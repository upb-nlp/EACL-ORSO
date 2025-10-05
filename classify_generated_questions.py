import json
import os


import wandb
import torch
from transformers import AutoTokenizer, ModernBertForSequenceClassification
from transformers import TrainingArguments, Trainer
import random
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
from utils import sorted_attributes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ATTRIBUTE = "local_or_sum" # focus local_or_sum

FILENAMES = [
    "EACL-ORSO/final_classification/fairytaleqa_test_responses_llama32_1b_orso_focus_local_or_sum_attribute.json",
    "EACL-ORSO/final_classification/fairytaleqa_test_responses_llama32_1b_steerlm_focus_local_or_sum_attribute.json",
    #"EACL-ORSO/final_classification/fairytaleqa_test_responses_llama32_1b_after_clustering_orso_focus_local_or_sum_attribute.json",
    #"EACL-ORSO/final_classification/fairytaleqa_test_responses_llama32_1b_after_clustering_steerlm_focus_local_or_sum_attribute.json",
    #"EACL-ORSO/final_classification/fairytaleqa_test_responses_llama32_1b_second_orso_focus_local_or_sum_attribute.json",
    #"EACL-ORSO/final_classification/fairytaleqa_test_responses_llama32_1b_orso_focus_attribute.json",
    #"EACL-ORSO/final_classification/fairytaleqa_test_responses_llama32_1b_orso_focus_local_or_sum_attribute.json",
    #"EACL-ORSO/final_classification/fairytaleqa_test_responses_llama32_1b_orso_local_or_sum_attribute.json",
    #"EACL-ORSO/final_classification/fairytaleqa_test_responses_llama32_1b_steerlm_focus_attribute.json",
    #"EACL-ORSO/final_classification/fairytaleqa_test_responses_llama32_1b_steerlm_focus_local_or_sum_attribute.json",
    #"EACL-ORSO/final_classification/fairytaleqa_test_responses_llama32_1b_steerlm_local_or_sum_attribute.json",
    
]

set_labels = sorted_attributes[ATTRIBUTE]
num_labels = len(set_labels)
label2id = {label: i for i, label in enumerate(set_labels)}
id2label = {i: label for i, label in enumerate(set_labels)}

model_name = f"modernbert_fairytaleqa_{ATTRIBUTE}_attribute"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ["HF_TOKEN"])
model = ModernBertForSequenceClassification.from_pretrained(
    model_name,
    token=os.environ["HF_TOKEN"],
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
    problem_type="multi_label_classification",
).to(device)

for filename in FILENAMES:
    NAME = filename.split("/")[-1].split(".")[0]

    dataset = json.load(open(filename))

    predictions = []

    prompts = [f"Context: {example['content']}\nQuestion: {example['question']}\nAnswer: {example['answer']}" if example['question'] and example['answer'] else f"none" for example in dataset]
    batch_size = 64
    for i in tqdm(range(0, len(prompts), batch_size)):
        end_idx = min(i + batch_size, len(prompts))
        batch_prompts = prompts[i:end_idx]
        inputs = tokenizer(batch_prompts, return_tensors="pt", max_length=4000, truncation=True, padding='max_length').to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()

        for j in range(len(batch_prompts)):
            pred = probs[j]
            pred_label = np.argmax(pred)
            pred_label = id2label[pred_label]
            predictions.append(pred_label)

    for p, d in zip(predictions, dataset):
        if not d['question'] or not d['answer']:
            p = "NO RESPONSE"
        d[f"predicted_{ATTRIBUTE}"] = p

    json.dump(dataset, open(f"EACL-ORSO/final_classification/{NAME}.json", "w"), indent=4)
