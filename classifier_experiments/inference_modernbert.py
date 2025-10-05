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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = json.load(open("EACL-ORSO/reshaped_datasets/fairytaleqa_test.json"))
data_train = json.load(open("EACL-ORSO/reshaped_datasets/fairytaleqa_train.json"))

ATTRIBUTE = "local_or_sum" # focus local_or_sum

set_labels = list(sorted(list(set([d[ATTRIBUTE] for d in data_train]))))
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

predictions = []

prompts = [f"Context: {example['content']}\nQuestion: {example['question']}\nAnswer: {example['answer']}" for example in dataset]
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
    d[f"predicted_{ATTRIBUTE}"] = p

json.dump(dataset, open(f"EACL-ORSO/classifier_experiments/fairytaleqa_test_predictions_{ATTRIBUTE}.json", "w"), indent=4)
