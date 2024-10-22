import os
import re
import random
import json
import argparse
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import recall_score, precision_score, f1_score
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins.training_type import DDPPlugin
from pytorch_lightning.plugins import DeepSpeedPlugin
import matplotlib.pyplot as plt
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM,
    BartForConditionalGeneration,
    BatchEncoding,
    RagConfig,
    BartConfig,
    DPRConfig,
    RagSequenceForGeneration,
    RagTokenForGeneration,
    RagTokenizer,
    T5ForConditionalGeneration,
    T5Config,
)
from src.re_model import WarmupQualityEstimatorModel
from src.main_model import CustomRagSequenceForGeneration

global t5_tokenizer
global trained_model
global wandb_logger

class RagCustomModel(pl.LightningModule):
    def __init__(self, model, learning_rate, batch_size, n_docs, checkpoints_dirpath=None):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_docs = n_docs
        self.checkpoints_dirpath = checkpoints_dirpath


    def forward(self, 
                context_input_ids, 
                context_attention_mask, 
                qe_input_ids,
                qe_attention_mask,
                decoder_input_ids, 
                decoder_attention_mask, 
                labels, 
                doc_scores, 
                n_docs):
        
        output = self.model(context_input_ids = context_input_ids,
                            context_attention_mask = context_attention_mask,
                            qe_input_ids = qe_input_ids,
                            qe_attention_mask = qe_attention_mask,                            
                            labels = labels,          
                            decoder_input_ids = decoder_input_ids,  
                            doc_scores = doc_scores,
                            n_docs = n_docs,                            
                            reduce_loss=True)
        
        return output.loss, output.loss_sub_1, output.loss_sub_2, output.loss_sub_3, output.logits, output.doc_scores, output.prob_true
    
    def select_top_k_documents_and_scores(self, doc_score_calculated, context_input_ids, top_k=25):

        batch_size, n_docs = doc_score_calculated.shape
        seq_len = context_input_ids.shape[1]

        context_input_ids = context_input_ids.view(batch_size, n_docs, seq_len)

        top_k_indices = doc_score_calculated.topk(top_k, dim=1).indices

        top_k_context_input_ids = torch.gather(context_input_ids, 1, top_k_indices.unsqueeze(-1).expand(-1, -1, seq_len))

        top_k_scores = torch.gather(doc_score_calculated, 1, top_k_indices)

        return top_k_context_input_ids, top_k_scores
    
    def normalize_text(self, text):

        text = re.sub(r'^Answer: ', '', text)
        text = text.strip()
        text = text.lower()

        text = re.sub(r'\b(a|an|the)\b', '', text)

        text = re.sub(r'[^\w\s]', '', text)

        text = re.sub(r'\s+', ' ', text).strip()

        return text
    
    def calculate_label_averages(self, doc_labels, prob_true):
        if doc_labels.shape != prob_true.shape:
            raise ValueError("Shapes of doc_labels and prob_true must be the same.")

        true_mask = doc_labels == 1
        if true_mask.sum() == 0:
            prob_true_label = torch.tensor(1.0)
        else:
            prob_true_label = prob_true[true_mask].mean()

        false_mask = doc_labels == 0
        prob_false_label = prob_true[false_mask].mean()

        return prob_true_label, prob_false_label

    def calculate_val_acc(self, logit, doc_scores, batch_size, n_docs = 100):
        predicted_token_ids = torch.argmax(logit, dim=-1)
        eos_token_id = 1
        decoding_text_result = []
        for i in range(batch_size):            
            decoded_texts = []
            decode_result = {}
            doc_score = doc_scores[i:(i+1), :]
            doc_score = torch.nn.functional.softmax(doc_score, dim=-1)
            for batch in predicted_token_ids[i * n_docs : (i+1) * n_docs, :]:
                if eos_token_id in batch:
                    eos_matches = (batch == eos_token_id).nonzero(as_tuple=True)[0]
                    if len(eos_matches) > 0:
                        eos_index = eos_matches[0].item()
                    else:
                        eos_index = len(batch)
                    batch = batch[:eos_index]

                decoded_text = rag_tokenizer.decode(batch, skip_special_tokens=True)
                decoded_texts.append(decoded_text)
                
            doc_score = doc_score.squeeze(0).tolist()
            
            for score, text in zip(doc_score, decoded_texts):
                if text in decode_result:
                    decode_result[text] += score
                else:
                    decode_result[text] = score
            max_score_text = max(decode_result, key=decode_result.get)
            decoding_text_result.append(max_score_text)
            
        return decoding_text_result
    
    def calculate_doc_metrics(self, doc_labels, prob_true):
        predicted = (prob_true >= 0.5).int()

        doc_labels_flat = doc_labels.view(-1).cpu().numpy()
        predicted_flat = predicted.view(-1).cpu().numpy()

        recall = recall_score(doc_labels_flat, predicted_flat)
        precision = precision_score(doc_labels_flat, predicted_flat)
        f1 = f1_score(doc_labels_flat, predicted_flat)

        return recall, precision, f1
    
    def calculate_tp_fp_fn(self, doc_labels, prob_true):
        predicted = (prob_true >= 0.5).int()

        tp = torch.sum((predicted == 1) & (doc_labels == 1))
        fp = torch.sum((predicted == 1) & (doc_labels == 0))
        fn = torch.sum((predicted == 0) & (doc_labels == 1))

        return tp, fp, fn
    
    def training_step(self, batch, batch_idx, optimizer_idx=None):
        _context_input_ids = batch["context_input_ids"]
        _context_attention_mask = batch["context_attention_mask"]
        _qe_input_ids = batch["qe_input_ids"]
        _qe_attention_mask = batch["qe_attention_mask"]
        _decoder_input_ids = batch["decoder_input_ids"]
        _decoder_attention_mask = batch["decoder_attention_mask"]
        _label = batch["label_ids"]
        _doc_scores = batch["doc_score"]

        batch_size = _context_input_ids.shape[0]
        context_input_ids = _context_input_ids.reshape(batch_size * _context_input_ids.shape[1], -1)
        context_attention_mask = _context_attention_mask.reshape(batch_size * _context_attention_mask.shape[1], -1)
        qe_input_ids = _qe_input_ids.reshape(batch_size * _qe_input_ids.shape[1], -1)
        qe_attention_mask = _qe_attention_mask.reshape(batch_size * _qe_attention_mask.shape[1], -1)
        decoder_input_ids = _decoder_input_ids.reshape(batch_size, -1)
        decoder_input_ids = self.model.rag.generator._shift_right(decoder_input_ids)
        decoder_attention_mask = _decoder_attention_mask.reshape(batch_size, -1)
        label = _label.reshape(batch_size, -1)  
        label = self.model.rag.generator._shift_right(label)
        doc_scores = _doc_scores.squeeze(1)  
        
            
        loss, loss_sub_1, loss_sub_2, loss_sub_3, output, doc_scores, prob_true = self(context_input_ids = context_input_ids,
                                                                                context_attention_mask = context_attention_mask,
                                                                                qe_input_ids = qe_input_ids,
                                                                                qe_attention_mask = qe_attention_mask,
                                                                                labels = label,                       
                                                                                decoder_input_ids = decoder_input_ids,     
                                                                                doc_scores = doc_scores,
                                                                                decoder_attention_mask = decoder_attention_mask,
                                                                                n_docs = self.n_docs)        
             
        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=self.batch_size)
        if args.loss_option == "yes":
            self.log("train_loss_1", loss_sub_1, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("train_loss_2", loss_sub_2, prog_bar=True, logger=True, batch_size=self.batch_size)
            self.log("train_loss_3", loss_sub_3, prog_bar=True, logger=True, batch_size=self.batch_size)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _context_input_ids = batch["context_input_ids"] 
        _context_attention_mask = batch["context_attention_mask"] 
        _qe_input_ids = batch["qe_input_ids"] 
        _qe_attention_mask = batch["qe_attention_mask"] 
        _decoder_input_ids = batch["decoder_input_ids"]
        _decoder_attention_mask = batch["decoder_attention_mask"]
        _label = batch["label_ids"] 
        _doc_scores = batch["doc_score"]
        doc_labels = batch["doc_labels"] 
        answers = batch["answers"]


        answers_arr = [[] for _ in range(len(answers[0]))]  
        for t in answers:
            for i, element in enumerate(t):
                answers_arr[i].append(element)
        
        
        batch_size = _context_input_ids.shape[0]
        context_input_ids = _context_input_ids.reshape(batch_size * _context_input_ids.shape[1], -1)
        context_attention_mask = _context_attention_mask.reshape(batch_size * _context_attention_mask.shape[1], -1)
        qe_input_ids = _qe_input_ids.reshape(batch_size * _qe_input_ids.shape[1], -1)
        qe_attention_mask = _qe_attention_mask.reshape(batch_size * _qe_attention_mask.shape[1], -1)
        decoder_input_ids = _decoder_input_ids.reshape(batch_size, -1)
        decoder_input_ids = self.model.rag.generator._shift_right(decoder_input_ids)
        decoder_attention_mask = _decoder_attention_mask.reshape(batch_size, -1)
        label = _label.reshape(batch_size, -1)   
        label = self.model.rag.generator._shift_right(label)
        doc_scores = _doc_scores.squeeze(1)
               
        loss, loss_sub_1, loss_sub_2, loss_sub_3, output, doc_scores, prob_true = self(context_input_ids = context_input_ids,
                                                                                        context_attention_mask = context_attention_mask,
                                                                                        qe_input_ids = context_input_ids,
                                                                                        qe_attention_mask = context_attention_mask,
                                                                                        labels = label,                       
                                                                                        decoder_input_ids = decoder_input_ids,     
                                                                                        doc_scores = doc_scores,
                                                                                        decoder_attention_mask = decoder_attention_mask,
                                                                                        n_docs = 100)
        self.log("val_loss", loss, prog_bar=True, logger=True, batch_size=self.batch_size)
        
        topk = 10
        context_input_ids_for_gen, doc_scores_for_gen = self.select_top_k_documents_and_scores(doc_scores, context_input_ids, topk)
        context_attention_mask_for_gen, doc_scores_for_gen = self.select_top_k_documents_and_scores(doc_scores, context_attention_mask, topk)
        context_input_ids_for_gen = context_input_ids_for_gen.reshape(batch_size * topk, -1)   
        context_attention_mask_for_gen = context_attention_mask_for_gen.reshape(batch_size * topk, -1)   
        
        
        generated_ids = self.model.generate(context_input_ids=context_input_ids_for_gen,
                                            context_attention_mask=context_attention_mask_for_gen,
                                            qe_input_ids = None, 
                                            qe_attention_mask = None, 
                                            doc_scores = doc_scores_for_gen, 
                                            n_docs = topk,
                                            num_beams=1, 
                                            max_length=25, repetition_penalty=2.5,
                                            do_deduplication = True,
                                            length_penalty=1,
                                            use_cache=True)   
        
        decoded_texts = rag_tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)  
        
        correct = sum([any(self.normalize_text(_pred) == self.normalize_text(_ans) for _ans in _answers) for _pred, _answers in zip(decoded_texts, answers_arr)])
        accuracy = correct / len(decoded_texts)

        self.log("val_accuracy", accuracy, prog_bar=True, logger=True, batch_size=self.batch_size)       
        
        
        tp, fp, fn = self.calculate_tp_fp_fn(doc_labels, prob_true)
        self.log("val_true_positive", tp.float(), prog_bar=True, logger=True, batch_size=self.batch_size, reduce_fx=torch.sum)
        self.log("val_false_positive", fp.float(), prog_bar=True, logger=True, batch_size=self.batch_size, reduce_fx=torch.sum)
        self.log("val_false_negative", fn.float(), prog_bar=True, logger=True, batch_size=self.batch_size, reduce_fx=torch.sum)  

        prob_true_label, prob_false_label = self.calculate_label_averages(doc_labels, prob_true)
        self.log("val_true_label_true_prob", prob_true_label, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log("val_false_label_true_prob", prob_false_label, prog_bar=True, logger=True, batch_size=self.batch_size)  

        return loss
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        _context_input_ids = batch["context_input_ids"]
        _context_attention_mask = batch["context_attention_mask"]
        _decoder_input_ids = batch["decoder_input_ids"]
        _decoder_attention_mask = batch["decoder_attention_mask"]
        _label = batch["label_ids"]
        _doc_scores = batch["doc_score"]
        
        batch_size = _context_input_ids.shape[0]
        context_input_ids = _context_input_ids.reshape(batch_size * _context_input_ids.shape[1], -1)
        context_attention_mask = _context_attention_mask.reshape(batch_size * _context_attention_mask.shape[1], -1)
        decoder_input_ids = _decoder_input_ids.reshape(batch_size, -1)
        decoder_input_ids = self.model.rag.generator._shift_right(decoder_input_ids)
        decoder_attention_mask = _decoder_attention_mask.reshape(batch_size, -1)
        label = _label.reshape(batch_size, -1)  
        label = self.model.rag.generator._shift_right(label)
        doc_scores = _doc_scores.reshape(batch_size, 100)    
        
        loss, loss_sub_1, loss_sub_2, loss_sub_3, output, doc_scores, prob_true = self(context_input_ids = context_input_ids,
                                                context_attention_mask = context_attention_mask,
                                                labels = label,                       
                                                decoder_input_ids = decoder_input_ids,     
                                                doc_scores = doc_scores,
                                                decoder_attention_mask = decoder_attention_mask,
                                                n_docs = 100)
        self.log("test_loss", loss, prog_bar=True, logger=True, batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        print("train_lr_equal")
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-3)
    
    
    
class RagQADataset(Dataset):
    def __init__(self, data, tokenizer, source_max_token_len=256, target_max_token_len=32, score_option="label", top_n = 20):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.score_option = score_option
        self.top_n = top_n

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        
        question = data_row.question
        answers = data_row.answers
        answers = answers[:512]
        while len(answers) < 512:
            answers.append('#$%*')

        answer = "Answer: " + data_row.target
        
        ctxs = data_row.ctxs
        
        _input_ids = self.tokenizer(question, return_tensors="pt", truncation=True, 
                                   return_attention_mask=True, add_special_tokens=True, padding="max_length")
        _label_ids = self.tokenizer(answer, return_tensors="pt", max_length = self.target_max_token_len, truncation=True, 
                                   return_attention_mask=True, add_special_tokens=True, padding="max_length")
        _decoder_ids = self.tokenizer(answer, return_tensors="pt", max_length = self.target_max_token_len, truncation=True, 
                                   return_attention_mask=True, add_special_tokens=False, padding="max_length")

        input_ids = _input_ids['input_ids']
        label_ids = _label_ids['input_ids']
        
        decoder_input_ids = _decoder_ids['input_ids']
        decoder_attention_mask = _decoder_ids['attention_mask']
        
        doc_cat = []
        doc_labels = []
        doc_score = []
        for doc in ctxs:
            doc_merge = "question: " + question + "context: " + doc['title'] + "." + doc['text']
            try:
                doc_label = str(doc['has_answer'])
            except:
                doc_label = "True"
            doc_cat.append(doc_merge)
            doc_labels.append(doc_label)
            
            if self.score_option == "label":
                if doc['has_answer'] == True:
                    score = 100.0
                    doc_score.append(score)
                else:
                    score = 0.1
                    doc_score.append(score)
            elif self.score_option == "dpr":
                doc_score.append(1.0)
            

        doc_score = torch.tensor(doc_score)
        doc_score = doc_score.unsqueeze(0)
        
        context_input_ids = []
        context_attention_mask = []

        for doc in doc_cat:
            inputs = self.tokenizer(
                doc,
                return_tensors="pt",
                padding="max_length",  
                max_length=self.source_max_token_len,  
                add_special_tokens=True,
                truncation=True,
            )
            context_input_ids.append(inputs['input_ids'])
            context_attention_mask.append(inputs['attention_mask'])
            
        
        context_input_ids = torch.cat(context_input_ids, dim=0)
        context_attention_mask = torch.cat(context_attention_mask, dim=0)
        
        tensor_doc_labels = torch.tensor([1 if label == 'True' else 0 for label in doc_labels])
        
        
        return {"input_ids": input_ids, "context_input_ids" : context_input_ids, "context_attention_mask" : context_attention_mask,
                "label_ids": label_ids, "doc_score" : doc_score, "decoder_attention_mask" : decoder_attention_mask, "decoder_input_ids":decoder_input_ids,
                "qe_input_ids": context_input_ids, "qe_attention_mask": context_attention_mask, "doc_labels" : tensor_doc_labels, "answers" : answers}


class RagQAModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, tokenizer, batch_size=8, source_max_token_len=396, target_max_token_len=32, score_option="label", top_n = 20):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df 
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.score_option = score_option
        self.top_n = top_n

    def setup(self):
        self.train_dataset = RagQADataset(self.train_df, self.tokenizer, self.source_max_token_len,
                                          self.target_max_token_len, self.score_option, self.top_n)
        self.test_dataset = RagQADataset(self.test_df, self.tokenizer, self.source_max_token_len,
                                          self.target_max_token_len, self.score_option, self.top_n)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)


def get_data_for_fine_tuning(path, n_epochs, learning_rate,
                             s_max_token_len=396, batch_size=8):
    with open(path) as f:
        f_js = json.loads(f.read()) 
    train_df = pd.DataFrame(f_js)
    
    with open(path) as g:
        g_js = json.loads(g.read()) 
    val_df = pd.DataFrame(g_js)
        
    return train_df, val_df


def fine_tune(train_df, val_df, tokenizer, module, model, source_max_token_len, target_max_token_len,
              batch_size, n_epochs, checkpoints_dirpath, learning_rate, logger, status, score_option, top_n):

    data_module = module(train_df, val_df, tokenizer,
                         source_max_token_len=source_max_token_len,
                         target_max_token_len=target_max_token_len,
                         batch_size=batch_size,
                         score_option = score_option,
                         top_n = top_n)
    
    data_module.setup()
        
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoints_dirpath,
                                          filename="rag_custom_model-{epoch:02d}-{val_loss:.3f}-{val_accuracy:.3f}", save_top_k=3,
                                          verbose=True, monitor="val_accuracy", mode="max")

    if status != "empty":
        print("Continue to training!")
        model = RagCustomModel.load_from_checkpoint(checkpoint_path = status, model = model, model_name = config["reader_model_name"],
                                                    learning_rate = config["learning_rate"], batch_size = config["batch_size"], 
                                                    n_docs = top_n, checkpoints_dirpath=checkpoints_dirpath)
        trainer = pl.Trainer(callbacks=[checkpoint_callback], max_epochs=n_epochs, devices=4, accelerator="gpu",
                            strategy="ddp", progress_bar_refresh_rate=20, logger=logger, accumulate_grad_batches=8, deterministic=True, precision="bf16")
        trainer.fit(model, data_module, ckpt_path=status)
    else:        
        model = RagCustomModel(model=model, learning_rate=learning_rate, batch_size=batch_size, 
                               n_docs = top_n, checkpoints_dirpath=checkpoints_dirpath)
        trainer = pl.Trainer(callbacks=[checkpoint_callback], max_epochs=n_epochs, devices=4, accelerator="gpu",
                            strategy="ddp", progress_bar_refresh_rate=20, logger=logger, accumulate_grad_batches=8, precision="bf16")
        trainer.fit(model, data_module)


if __name__ == '__main__':
        
    f = open(os.getcwd() + "/config_gen.json")
    config = json.load(f)["training"]
    f = open(os.getcwd() + "/config_re.json")
    qe_config = json.load(f)["training"]
    os.environ["WANDB_API_KEY"] = config["wandb_api_key"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=False,
                        default="./dataset/",
                        help="data path prefix for training/val data")
    parser.add_argument("--wandb_name", type=str, required=True, help="wandb_project name")
    parser.add_argument("--model", type=str, required=False,
                        default="empty",
                        help="If you have trained model, then put trained model path")
    parser.add_argument("--qe_model", type=str, required=False,
                        default="empty",
                        help="If you have pretrained RE model, then put trained qe model path. Use default settings by default")    
    parser.add_argument("--score", type=str, required=False,
                        default="label",
                        help="You will only utilize this option if you are not using RE modules. This option is not automatically utilized when training RE modules (does not utilize any prior scores).")
    parser.add_argument("--top_n", type=int, required=False,
                        default=20,
                        help="Enter the total number of contexts in the test dataset")
    parser.add_argument("--checkpoints_dirpath", type=str, required=False,
                        default="results/",
                        help="Enter the path where the results are saved")
    parser.add_argument("--loss_option", type=str, required=False,
                        default="yes",
                        help="if you want prob contraint loss set this value yes")
    parser.add_argument("--loss_weight", type=float, required=False,
                        default=1,
                        help="hyperparameter of addtional loss(alpha 1)")
    args = parser.parse_args()
    
    wandb_logger = WandbLogger(project=args.wandb_name)


    retriever_model_name = config['retriever_model_name']
    reader_model_name = config['reader_model_name']
    
    question_encoder_config = DPRConfig()  
    generator_config = T5Config()  

    rag_config = RagConfig.from_question_encoder_generator_configs(
        question_encoder_config=question_encoder_config,
        generator_config=generator_config
    )

    rag_config.n_docs = args.top_n
    rag_config.retrieval_batch_size = config["batch_size"]
    

    question_encoder_tokenizer = AutoTokenizer.from_pretrained(retriever_model_name, cache_dir="tokenizer/dpr-encoder")
    generator_tokenizer = AutoTokenizer.from_pretrained(reader_model_name, cache_dir=f"tokenizer/{reader_model_name}")

    rag_tokenizer = RagTokenizer(question_encoder=question_encoder_tokenizer, generator=generator_tokenizer)

    rag_tokenizer._switch_to_target_mode()
    
    if args.qe_model == "empty":
        qe_model_name = qe_config['model_name']

        classifier_config = T5Config.from_pretrained(qe_model_name, cache_dir=f"config/{qe_model_name}") # Config for the classifier
        
        qe_gen_model = T5ForConditionalGeneration.from_pretrained(qe_model_name, config=classifier_config, cache_dir=f"models/{qe_model_name}")

        qe_model = WarmupQualityEstimatorModel(model=qe_gen_model, coef = qe_config["coef"], learning_rate = qe_config["learning_rate"], 
                                            batch_size = qe_config["batch_size"], loss_func = "ce", lr_control = "equal")
    else:
        qe_model_name = qe_config['model_name']
        
        classifier_config = T5Config.from_pretrained(qe_model_name, cache_dir=f"config/{qe_model_name}")
        
        qe_gen_model = T5ForConditionalGeneration.from_pretrained(qe_model_name, config=classifier_config, cache_dir=f"models/{qe_model_name}")
        
        qe_model = WarmupQualityEstimatorModel.load_from_checkpoint(args.qe_model, model = qe_gen_model, model_name = qe_model_name, coef = qe_config["coef"],
                                                        learning_rate = qe_config["learning_rate"], batch_size = qe_config["batch_size"], loss_func = "ce", lr_control = "equal")
    
    
    ########################## build RAG generator model ##########################
    generator_model = AutoModelForSeq2SeqLM.from_pretrained(reader_model_name, cache_dir=f"models/{reader_model_name}")

    model = CustomRagSequenceForGeneration(config=rag_config, generator=generator_model, quality_estimator=qe_model,
                                           loss_option=args.loss_option, loss_weight=args.loss_weight)
    
        
    ########################## To training ##########################    
    
    path = args.path
    train_df, val_df = get_data_for_fine_tuning(path=path,
                                                s_max_token_len=config["source_max_token_len"],
                                                batch_size=config["batch_size"],
                                                n_epochs=config["n_epochs"],
                                                learning_rate=config["learning_rate"])
    
    fine_tune(train_df = train_df, 
              val_df = val_df, 
              tokenizer = rag_tokenizer, 
              module = RagQAModule, 
              model = model,
              source_max_token_len=config["source_max_token_len"],
              target_max_token_len=config["target_max_token_len"],
              batch_size=config["batch_size"],
              n_epochs=config["n_epochs"],
              checkpoints_dirpath=args.checkpoints_dirpath,
              learning_rate=config["learning_rate"],
              logger=wandb_logger,
              status = args.model,
              score_option = args.score,
              top_n = args.top_n
              )
