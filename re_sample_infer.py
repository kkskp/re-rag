import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

def probability_to_logit(prob: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return torch.log(prob + eps) - torch.log((1 - prob) + eps)

def main():
    model_path = "nq/large"  

    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("t5-base", cache_dir="tokenizer")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    question = "who got the first nobel prize in physics"
    title = "Charles Russell Bardeen"
    text = "business before marrying, and was an active figure in the art world. After her death from cancer in 1921, Charles married Ruth Hames. His son, Dr. John Bardeen, became the only person to win the Nobel Prize in Physics twice, in 1956 and 1972. Bardeen died in Madison, Wisconsin in 1935, from pancreatic cancer. He was succeeded as Dean of the University of Wisconsin Medical School by Dr. William Shainline Middleton. Charles Russell Bardeen Charles Russell Bardeen (8 February 1871 – 12 June 1935) was an American physician and anatomist and the first dean of the University of Wisconsin Medical"
        
    prompt = f"question: {question}context: {title}. {text}"

    inputs = tokenizer(prompt, 
                       max_length=256,
                       padding="max_length",
                       truncation=True,
                       return_attention_mask=True,
                       add_special_tokens=True,
                       return_tensors="pt").to(device)

    # 생성
    with torch.no_grad():
        outputs = model.generate(
            input_ids = inputs["input_ids"],
            attention_mask = inputs["attention_mask"],
            max_length=1,  
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True
        )

    generated_seq = outputs.sequences[0]  
    decoded_text = tokenizer.decode(generated_seq, skip_special_tokens=True)

    print("\n=== Calculations for logit of true token and true token probability ===")

    scores_per_step = outputs.scores  

    for step, step_logits in enumerate(scores_per_step):
        logits = step_logits[0] 
        probabilities = torch.softmax(logits, dim=-1) 

        prob_true_raw = probabilities[1176]
        prob_false_raw = probabilities[6136]

        prob_sum = prob_true_raw + prob_false_raw
        prob_true = prob_true_raw / prob_sum

        logit_true = probability_to_logit(prob_true)
        doc_score_calculated = logit_true

        token_id = outputs.sequences[0][step+1]
        token_str = tokenizer.decode(token_id)

        print(
            f"[Step {step+1}] "
            f"Token: {token_str}, "
            f"true token logit: {doc_score_calculated.item():.6f}, "
            f"true token prob: {prob_true.item():.6f}"
        )


if __name__ == "__main__":
    main()