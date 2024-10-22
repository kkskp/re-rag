import os
import json
import time
import re
import torch
from training import RagCustomModel
from pathlib import Path
from tqdm import tqdm

from src.main_model import CustomRagSequenceForGeneration
from src.re_model import WarmupQualityEstimatorModel
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM,
    BartForConditionalGeneration,
    BatchEncoding,
    RagConfig,
    BartConfig,
    T5Config,
    DPRConfig,
    RagSequenceForGeneration,
    RagTokenForGeneration,
    RagTokenizer,
    T5ForConditionalGeneration,
)
import pandas as pd
import argparse


def select_top_k_documents_and_scores(doc_score_calculated, context_input_ids, top_k=25):

    batch_size, n_docs = doc_score_calculated.shape
    seq_len = context_input_ids.shape[1]

    context_input_ids = context_input_ids.view(batch_size, n_docs, seq_len)

    top_k_indices = doc_score_calculated.topk(top_k, dim=1).indices

    top_k_context_input_ids = torch.gather(context_input_ids, 1, top_k_indices.unsqueeze(-1).expand(-1, -1, seq_len))

    top_k_scores = torch.gather(doc_score_calculated, 1, top_k_indices)

    return top_k_context_input_ids, top_k_scores


def df_chunks_generator(df, n, input_col_1, input_col_2, score_option):
    for i in range(0, len(df), n):
        small_df = df.iloc[i:i + n]
        c = small_df[input_col_1].to_list()
        q = small_df[input_col_2].to_list()
        
        doc_cat = []
        doc_qe_cat = []
        doc_score = []
        for question in q:
            for item in c:
                for doc in item:
                    doc_merge = "question: " + question + "context: " + doc['title'] + "." + doc['text']
                    doc_cat.append(doc_merge)
                    
                    doc_qe_merge = "question: " + question + "context: " + doc['title'] + "." + doc['text']
                    doc_qe_cat.append(doc_qe_merge)
                    
                    if score_option == "label":
                        if doc['has_answer'] == True:
                            score = 100.0
                            doc_score.append(score)
                        else:
                            score = 0.1
                            doc_score.append(score)
                    elif score_option == "dpr":
                        doc_score.append(1.0)
                    elif score_option == "qc":
                        doc_score.append(1.0)
                        
        doc_score = torch.tensor(doc_score)
        doc_score = doc_score.unsqueeze(0)
                
        yield doc_cat, doc_qe_cat, doc_score


def generate_answer(question_and_context, qe_text, score, model, tokenizer, input_max_length, output_max_length, repetition_penalty, length_penalty,
                    num_beams, score_option, n_docs, top_k):

    source_encoding = tokenizer(question_and_context, max_length=input_max_length,
                                padding="max_length", truncation=True, return_attention_mask=True,
                                add_special_tokens=True, return_tensors="pt") # n_docs(100), seq_len
  
    
    source_encoding = source_encoding.to(device)
    score = score.to(device)
    with torch.no_grad():
        if score_option == "qc_cal":
            doc_score_calculated, prob_true, prob_true_raw_mean, prob_false_raw_mean, prob_false_raw, prob_true_raw = model.model.calculate_qc_docscore(source_encoding["input_ids"], source_encoding["attention_mask"], n_docs, temperature = 1.0)

            topk = top_k
            context_input_ids_for_gen, doc_scores_for_gen = select_top_k_documents_and_scores(doc_score_calculated, source_encoding["input_ids"], topk)
            context_attention_mask_for_gen, doc_scores_for_gen = select_top_k_documents_and_scores(doc_score_calculated, source_encoding["attention_mask"], topk)
            context_input_ids_for_gen = context_input_ids_for_gen.reshape(1 * topk, -1)   
            context_attention_mask_for_gen = context_attention_mask_for_gen.reshape(1 * topk, -1)  

            generated_ids = model.model.generate(context_input_ids = context_input_ids_for_gen,
                                                context_attention_mask = context_attention_mask_for_gen,
                                                qe_input_ids = None, # formmat unified
                                                qe_attention_mask = None, # formmat unified
                                                doc_scores = doc_scores_for_gen, 
                                                n_docs = context_input_ids_for_gen.shape[0],
                                                num_beams=num_beams, 
                                                max_length=output_max_length, repetition_penalty=repetition_penalty,
                                                do_deduplication = True,
                                                length_penalty=length_penalty,
                                                use_cache=True)
        else:
            topk = top_k
            context_input_ids_for_gen, doc_scores_for_gen = select_top_k_documents_and_scores(score, source_encoding["input_ids"], topk)
            context_attention_mask_for_gen, doc_scores_for_gen = select_top_k_documents_and_scores(score, source_encoding["attention_mask"], topk)
            context_input_ids_for_gen = context_input_ids_for_gen.reshape(1 * topk, -1)   
            context_attention_mask_for_gen = context_attention_mask_for_gen.reshape(1 * topk, -1)  
        
            generated_ids = model.model.generate(context_input_ids=context_input_ids_for_gen,
                                                context_attention_mask=context_attention_mask_for_gen,
                                                qe_input_ids = None, # formmat unified
                                                qe_attention_mask = None, # formmat unified
                                                doc_scores = doc_scores_for_gen, 
                                                n_docs = context_input_ids_for_gen.shape[0],
                                                num_beams=num_beams, 
                                                max_length=output_max_length, repetition_penalty=repetition_penalty,
                                                do_deduplication = True,
                                                length_penalty=length_penalty,
                                                use_cache=True)
        
    preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return preds


def query_model(model, path, checkpoint_name, tokenizer, config_dict, score_option, partition, column_option, n_docs, top_k):

    with open(path) as f:
        f_js = json.loads(f.read()) 
    df = pd.DataFrame(f_js)
    
    if partition == 100:
        df = df
    else:
        df = df.iloc[partition * 1000 : (partition + 1) * 1000]
    
    vals = []
    for q_c, qe_text, score in tqdm(df_chunks_generator(df, config_dict["batch_size"], column_option, "question", score_option)):
        model_output = generate_answer(q_c,
                                       qe_text,
                                       score,  # pass in batches
                                       model = model,
                                       tokenizer = tokenizer,
                                       input_max_length=config_dict['input_max_length'],
                                       output_max_length=config_dict['output_max_length'],
                                       repetition_penalty=config_dict['repetition_penalty'],
                                       length_penalty=config_dict['length_penalty'],
                                       num_beams=config_dict['num_beams'],
                                       score_option=score_option,
                                       n_docs = n_docs,
                                       top_k = top_k)
        vals.extend(model_output)
        print(model_output)
    df = df.assign(model_output=vals)
    save_results_to_file(df, checkpoint_name, config_dict, path, partition)


def save_results_to_file(df, checkpoint_name, config_dict, original_path, partition):
    checkpoint_name = re.sub(".*checkpoints/", "", checkpoint_name)
    results_dir, model_name = config_dict['results_dir'], config_dict['model_name']
    file_name = f"{results_dir}/{model_name}/{checkpoint_name}_inference_{partition}.csv"
    Path(os.path.dirname(file_name)).mkdir(parents=True, exist_ok=True)
    df.to_csv(file_name)





if __name__ == '__main__':
    ########################## build arguments and main config ##########################
    
    f = open(os.getcwd() + "/config_gen.json")
    config = json.load(f)["training"]
    f = open(os.getcwd() + "/config_gen.json")
    config_query = json.load(f)["evaluation"]
    f = open(os.getcwd() + "/config_re.json")
    qe_config = json.load(f)["training"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                        default="./dataset/",
                        help="Data path prefix for test data")
    parser.add_argument("--model", type=str, required=False,
                        default="empty",
                        help="Trained model path")
    parser.add_argument("--checkpoint_name", type=str, required=True,
                        help="Enter your checkpoint name. The name entered is used as the output filename")
    parser.add_argument("--score_option", type=str, required=True,
                        default="qc_cal",
                        help="The default setting is qc_cal, from which the score is calculated by the RE module. For other score usage, set it to custom.")
    parser.add_argument("--top_n", type=int, required=False,
                        default=100,
                        help="Enter the total number of contexts in the test dataset")
    parser.add_argument("--top_k", type=int, required=False,
                        default=25,
                        help="Enter the number of top-k contexts to use")
    parser.add_argument("--qe_model", type=str, required=False,
                        default="empty",
                        help="Choose RE model option, use default option")
    parser.add_argument("--select_column", type=str, required=False,
                        default="ctxs",
                        help="Choose ctxs, ctxs_5, ctxs_10, ctxs_20, ctxs_25, ctxs_50")
    parser.add_argument("--eval_partition", type=int, required=False,
                        default=0,
                        help="Divide eval inference, give number 0 ~ 3")
    args = parser.parse_args()
    

    ########################## build tokenizer ##########################

    retriever_model_name = config['retriever_model_name']
    reader_model_name = config['reader_model_name']
    
    question_encoder_config = DPRConfig() 
    generator_config = T5Config() 

    rag_config = RagConfig.from_question_encoder_generator_configs(
        question_encoder_config=question_encoder_config,
        generator_config=generator_config
    )  
    rag_config.generator.early_stopping = True 
    rag_config.generator.n_docs = args.top_n
    
    question_encoder_tokenizer = AutoTokenizer.from_pretrained(retriever_model_name, cache_dir="tokenizer/dpr-encoder")
    generator_tokenizer = AutoTokenizer.from_pretrained(reader_model_name, cache_dir="tokenizer/t5-small")

    rag_tokenizer = RagTokenizer(question_encoder=question_encoder_tokenizer, generator=generator_tokenizer)

    rag_tokenizer._switch_to_target_mode()
    
    ########################## build RAG model ##########################
    
    generator_model = AutoModelForSeq2SeqLM.from_pretrained(reader_model_name, cache_dir="models/t5-small")
    
    if args.qe_model == "empty":
        qe_model_name = qe_config['model_name']
        classifier_config = T5Config.from_pretrained(qe_model_name, cache_dir=f"config/{qe_model_name}") 
        
        qe_gen_model = T5ForConditionalGeneration.from_pretrained(qe_model_name, config=classifier_config, cache_dir=f"models/{qe_model_name}")

        qe_model = WarmupQualityEstimatorModel(model=qe_gen_model, coef = qe_config["coef"], learning_rate = qe_config["learning_rate"], 
                                            batch_size = qe_config["batch_size"], loss_func = "ce", lr_control = "equal")
        
        rag_model = CustomRagSequenceForGeneration(config=rag_config, generator=generator_model, quality_estimator=qe_model)
    
    else:
        qe_model_name = qe_config['model_name']
    
        classifier_config = T5Config.from_pretrained(qe_model_name, cache_dir=f"config/{qe_model_name}")
        
        qe_gen_model = T5ForConditionalGeneration.from_pretrained(qe_model_name, config=classifier_config, cache_dir=f"models/{qe_model_name}")
        
        qe_model = WarmupQualityEstimatorModel.load_from_checkpoint(args.qe_model, model = qe_gen_model, model_name = qe_model_name, coef = qe_config["coef"],
                                                        learning_rate = qe_config["learning_rate"], batch_size = qe_config["batch_size"], loss_func = "ce", lr_control = "equal")
        
        rag_model = CustomRagSequenceForGeneration(config=rag_config, generator=generator_model, quality_estimator=qe_model)

    if args.model == "empty":
        print("run pretrained model")
        model = rag_model
    else:
        print("run finetuned model")
        model = RagCustomModel.load_from_checkpoint(args.model, model = rag_model, model_name = config["reader_model_name"],
                                                    learning_rate = config["learning_rate"], batch_size = config["batch_size"], n_docs = args.top_n)
                                
    
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.model.eval()
    model = model.to(device) 
    
    tic = time.perf_counter()
    query_model(model = model,
                path=args.data,
                tokenizer = rag_tokenizer,
                checkpoint_name=args.checkpoint_name,
                config_dict=config_query,
                score_option = args.score_option,
                partition = args.eval_partition,
                column_option = args.select_column,
                n_docs = args.top_n,
                top_k = args.top_k)
    toc = time.perf_counter()
    print(f"Running time: {toc - tic:0.4f} seconds")