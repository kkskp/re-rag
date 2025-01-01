import os
import json
import time
import re

import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from tqdm import tqdm
import openai
from openai import OpenAI

from transformers import T5ForConditionalGeneration, AutoTokenizer

def probability_to_logit(prob: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return torch.log(prob + eps) - torch.log((1 - prob) + eps)

def load_t5_model(model_path: str = "nq/large", tokenizer_name: str = "t5-base"):
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir="tokenizer")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device

def add_true_token_scores_to_df(df: pd.DataFrame, model: T5ForConditionalGeneration, tokenizer: AutoTokenizer, device: torch.device, save_path: str = "updated_data.json"):
    true_token_id = 1176
    false_token_id = 6136

    for idx in tqdm(range(len(df)), desc="(1) add_true_token_scores"):
        question = df.iloc[idx]["question"]
        ctxs_list = df.iloc[idx]["ctxs"]

        if not isinstance(ctxs_list, list):
            continue

        for ctx_dict in ctxs_list:
            title = ctx_dict.get("title", "")
            text = ctx_dict.get("text", "")
            prompt = f"question: {question}context: {title}. {text}"

            inputs = tokenizer(
                prompt,
                max_length=256,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=1,  
                    num_beams=1,
                    return_dict_in_generate=True,
                    output_scores=True
                )

            scores_per_step = outputs.scores

            if len(scores_per_step) == 0:
                ctx_dict["re_score"] = float(0.0)
                ctx_dict["true_prob"] = float(0.0)
                continue

            last_step_logits = scores_per_step[-1][0]
            probabilities = torch.softmax(last_step_logits, dim=-1)
            prob_true_raw = probabilities[true_token_id]
            prob_false_raw = probabilities[false_token_id]
            prob_sum = prob_true_raw + prob_false_raw

            if prob_sum.item() < 1e-15:
                prob_true = torch.tensor(0.0, device=device)
            else:
                prob_true = prob_true_raw / prob_sum

            logit_true = probability_to_logit(prob_true)

            ctx_dict["re_score"] = float(logit_true.item())
            ctx_dict["true_prob"] = float(prob_true.item())

    df.to_json(save_path, orient="records", force_ascii=False)
    return df

def sort_ctxs_score(row):
    sorted_ctxs = sorted(row.ctxs, key=lambda x: x.get('qc_score', 0.0), reverse=True)
    return sorted_ctxs

def make_prompt(question, context):
    prompt = f"""
### Description : Below are some examples of question and answer formats. Use these examples as a guide to help you come up with the right answer to the question you'll eventually be asked.

- Example 1
Context:
'title': Sports in the United States,
'text': Erving (won MVP awards in both the ABA and NBA), Kareem Abdul-Jabbar (6 time MVP), ...
Question: who are the top 5 leading scorers in nba histor
Answer: Kobe Bryant

(...Your Examples...)

### Instructions: Provide the correct answer to the given question. The given question is accompanied by context related to the question. Be sure to refer to the context provided and enter your answer based on the context after "###Answer:". The question and the answer you provide must be relevant. Write your answer in a short "short answer" of 5 words or less. 

### Context: 
{context}

### Question: {question}

### Answer:
    """
    return prompt

def make_context_set(context, topk, option="sep"):
    context_extract = context[:topk]
    context_score = []
    if option == "sep":
        context_set = []
        for i in range(len(context_extract)):
            t = context_extract[i].get('title', '')
            c = context_extract[i].get('text', '')
            context_set.append([f"title: {t}, text: {c}"])
            context_score.append(context_extract[i].get('qc_score', 0.0))
    else:
        context_set = ""
        for i in range(len(context_extract)):
            t = context_extract[i].get('title', '')
            c = context_extract[i].get('text', '')
            context_set = "\n".join([context_set, f"context {i} title: {t}. text: {c}"])
            context_score.append(context_extract[i].get('qc_score', 0.0))
    return context_set, context_score

def get_answer_and_probabilities(question, client, model="gpt-3.5-turbo-0125"):
    params = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 32,
        "temperature": 0.8,
        "stop": None,
        "seed": 123,
        "logprobs": True,
        "top_logprobs": 1,
    }
    response = client.chat.completions.create(**params)
    return response

def generate_final_answers(df: pd.DataFrame, save_path: str = "final_result.json"):
    df['ctxs'] = df.apply(sort_ctxs_score, axis=1)
    df['gpt_answer_qc_10'] = pd.NA
    df['gpt_answer_qc_5'] = pd.NA
    df.reset_index(drop=True, inplace=True)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-xxxxx"))
    NUM_INPUT_CONTEXT = 10
    NUM_INPUT_CONTEXT_CUT = 5

    for i in tqdm(range(len(df)), desc="(2) generate_final_answers"):
        dic_10 = {}
        dic_5 = {}
        question = df.iloc[i].question
        context = df.iloc[i].ctxs
        context_set, context_score = make_context_set(context, NUM_INPUT_CONTEXT, "sep")
        tensor_10 = torch.tensor(context_score, dtype=torch.float32)
        softmax_scores_10 = F.softmax(tensor_10, dim=0)
        tensor_5 = torch.tensor(context_score[:NUM_INPUT_CONTEXT_CUT], dtype=torch.float32)
        softmax_scores_5 = F.softmax(tensor_5, dim=0)

        for j in range(NUM_INPUT_CONTEXT):
            concat_text = make_prompt(question, context_set[j])
            response = get_answer_and_probabilities(concat_text, client=client)
            answer_sequence = response.choices[0].message.content

            if hasattr(response.choices[0], "logprobs") and response.choices[0].logprobs:
                top_logprobs = response.choices[0].logprobs["token_logprobs"]
            else:
                top_logprobs = []

            answer_likelihood = 1.0
            for k in range(len(top_logprobs)):
                log_prob = top_logprobs[k]
                prob = np.exp(log_prob) if log_prob is not None else 1.0
                answer_likelihood *= prob

            if j < NUM_INPUT_CONTEXT_CUT:
                answer_likelihood_10 = answer_likelihood * softmax_scores_10[j].item()
                answer_likelihood_5 = answer_likelihood * softmax_scores_5[j].item()
                dic_10[answer_sequence] = dic_10.get(answer_sequence, 0) + answer_likelihood_10
                dic_5[answer_sequence] = dic_5.get(answer_sequence, 0) + answer_likelihood_5
            else:
                answer_likelihood_10 = answer_likelihood * softmax_scores_10[j].item()
                dic_10[answer_sequence] = dic_10.get(answer_sequence, 0) + answer_likelihood_10

        if len(dic_10) > 0:
            answer_10 = max(dic_10, key=dic_10.get)
            df.loc[i, 'gpt_answer_qc_10'] = answer_10
        if len(dic_5) > 0:
            answer_5 = max(dic_5, key=dic_5.get)
            df.loc[i, 'gpt_answer_qc_5'] = answer_5

    df.to_json(save_path, orient="records", force_ascii=False)
    return df

def main():
    input_json_path = "input_data.json"
    df = pd.read_json(input_json_path, orient="records")
    model, tokenizer, device = load_t5_model(
        model_path="nq/large",
        tokenizer_name="t5-base"
    )
    df_updated = add_true_token_scores_to_df(
        df=df,
        model=model,
        tokenizer=tokenizer,
        device=device,
        save_path="updated_data.json"
    )
    df_final = generate_final_answers(
        df=df_updated,
        save_path="final_result.json"
    )
    print("Done. Final result saved to 'final_result.json'.")

if __name__ == "__main__":
    main()
