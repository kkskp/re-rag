import json
import os
import pandas as pd
import torch
import random
import copy
import gc
import argparse
from tqdm import tqdm
import pytorch_lightning as pl

from transformers import DPRConfig, RagConfig, AutoTokenizer, RagTokenizer, T5ForConditionalGeneration, T5Config, AutoModelForSeq2SeqLM
from src.main_model import CustomRagSequenceForGeneration
from src.re_model import WarmupQualityEstimatorModel
from training import RagCustomModel

torch.cuda.empty_cache()
pl.seed_everything(42)
random.seed(42)

def cal_dpr_score(df, model, rag_tokenizer, device, temperature):
    model = model.to(device)
    model.eval()
    num_rows = df.shape[0]
    for i in tqdm(range(num_rows), desc="Get Confidence Data"):
        question = df.iloc[i]['question']
        context = []
        for j in range(len(df.iloc[i]['ctxs'])):
            context_merge = "Query:" + question + "Document:" + df.iloc[i]['ctxs'][j]['title'] + ". " + df.iloc[i]['ctxs'][j]['text'] + "Relevant:"
            context.append(context_merge)
        context_inputs = rag_tokenizer(context, max_length=256, return_tensors='pt', padding="max_length", truncation=True)
        context_inputs = context_inputs.to(device)

        with torch.no_grad():
            if args.model == "empty":
                qc_docscore, probs, prob_true_raw_mean, prob_false_raw_mean, prob_false_raw, prob_true_raw = model.calculate_qc_docscore(context_inputs['input_ids'], context_inputs['attention_mask'], len(df.iloc[i]['ctxs']), temperature)
            else:
                qc_docscore, probs, prob_true_raw_mean, prob_false_raw_mean, prob_false_raw, prob_true_raw = model.model.calculate_qc_docscore(context_inputs['input_ids'], context_inputs['attention_mask'], len(df.iloc[i]['ctxs']), temperature)
        for k in range(qc_docscore.shape[1]):            
            df.iloc[i]['ctxs'][k]['prob_true'] = probs[:,k].item()
            df.iloc[i]['ctxs'][k]['qc_score'] = qc_docscore[:,k].item()

        del context_inputs
        torch.cuda.empty_cache()
        gc.collect()


    return df


if __name__ == '__main__':

    f = open(os.getcwd() + "/config_gen.json")
    config = json.load(f)["fine_tune"]
    f = open(os.getcwd() + "/config_re.json")
    qe_config = json.load(f)["fine_tune"]
    os.environ["WANDB_API_KEY"] = config["wandb_api_key"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False,
                        default="empty",
                        help="data path prefix for test data")
    parser.add_argument("--qe_model", type=str, required=False,
                        default="empty",
                        help="data path prefix for test data")
    parser.add_argument("--temperature", type=float, required=False,
                        default=1.0,
                        help="data path prefix for test data")
    args = parser.parse_args()
    print(args.model)

    temperature = args.temperature

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    retriever_model_name = "facebook/dpr-question_encoder-single-nq-base"
    reader_model_name = config['reader_model_name']  
    question_encoder_config = DPRConfig()  
    generator_config = T5Config()  
    rag_config = RagConfig.from_question_encoder_generator_configs(
        question_encoder_config=question_encoder_config,
        generator_config=generator_config
    )
        

    question_encoder_tokenizer = AutoTokenizer.from_pretrained(retriever_model_name, cache_dir="tokenizer/dpr-encoder")
    generator_tokenizer = AutoTokenizer.from_pretrained(reader_model_name, cache_dir="tokenizer/t5-small")

    rag_tokenizer = RagTokenizer(question_encoder=question_encoder_tokenizer, generator=generator_tokenizer)

    rag_tokenizer._switch_to_target_mode()

    qe_model_name = qe_config["model_name"]
    classifier_config = T5Config.from_pretrained(qe_model_name, cache_dir=f"config/{qe_model_name}")    
    qe_gen_model = T5ForConditionalGeneration.from_pretrained(qe_model_name, config=classifier_config, cache_dir=f"models/{qe_model_name}")    
    
    if args.qe_model == "empty":
        qe_model = WarmupQualityEstimatorModel(model=qe_gen_model, coef = qe_config["coef"], learning_rate = qe_config["learning_rate"], 
                                            batch_size = qe_config["batch_size"], loss_func = "ce", lr_control = "equal") 
    else:
        qe_model = WarmupQualityEstimatorModel.load_from_checkpoint(args.qe_model, model = qe_gen_model, model_name = qe_model_name, coef = qe_config["coef"],
                                                    learning_rate = qe_config["learning_rate"], batch_size = qe_config["batch_size"], loss_func = "ce", lr_control = "equal")    
    
    generator_model = AutoModelForSeq2SeqLM.from_pretrained(reader_model_name, cache_dir="models/t5-small")
    rag_model = CustomRagSequenceForGeneration(config=rag_config, generator=generator_model, quality_estimator=qe_model)

    if args.model != "empty":
        print("run finetuned model")
        model = RagCustomModel.load_from_checkpoint(args.model, model = rag_model, model_name = config["reader_model_name"],
                                            learning_rate = config["learning_rate"], batch_size = config["batch_size"], n_docs = 20)
    else:
        print("run pretrained model")
        model = rag_model

    with open("your_dataset_path") as f:
        js = json.loads(f.read())
    df_train = pd.DataFrame(js)
    
    df_train = cal_dpr_score(df_train, model, rag_tokenizer, device)
    
    df_train.to_json(f"your_dataset_path", orient='columns')
