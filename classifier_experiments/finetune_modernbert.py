import json
import os

import wandb
import torch
from transformers import AutoTokenizer, ModernBertForSequenceClassification
from transformers import TrainingArguments, Trainer
import random
import numpy as np
from sklearn.metrics import f1_score
from datasets import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ATTRIBUTE = "local_or_sum" # focus local_or_sum

new_model_name = f"modernbert_fairytaleqa_{ATTRIBUTE}_attribute"
model_name = "answerdotai/ModernBERT-large"

wandb.login(key=os.environ["WANDB_TOKEN"])
wandb.init(
    project="ORSO",
    config={
        'task': f"FairytaleQA {ATTRIBUTE} Attribute Classification",
        'base_model': model_name,
        'model_name': new_model_name,
    }
)

data_train = json.load(open("EACL-ORSO/reshaped_datasets/fairytaleqa_train.json"))
data_val = json.load(open("EACL-ORSO/reshaped_datasets/fairytaleqa_val.json"))

random.shuffle(data_train)

# Convert to dataset
train_dataset = Dataset.from_list(data_train)
test_dataset = Dataset.from_list(data_val)

tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ["HF_TOKEN"])

set_labels = list(sorted(list(set([d[ATTRIBUTE] for d in data_train]))))
num_labels = len(set_labels)
label2id = {label: i for i, label in enumerate(set_labels)}
id2label = {i: label for i, label in enumerate(set_labels)}

def tokenize_function(example):
    prompt = f"Context: {example['content']}\nQuestion: {example['question']}\nAnswer: {example['answer']}"
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=4000, truncation=True, padding='max_length').to(device)
    inputs["attention_mask"] = inputs["attention_mask"].squeeze()
    inputs["input_ids"] = inputs["input_ids"].squeeze()
    inputs["no_tokens"] = inputs["input_ids"].shape[0]

    correct_label_id = label2id[example[ATTRIBUTE]]
    correct_labels_tensor = torch.zeros(num_labels)
    correct_labels_tensor[correct_label_id] = 1
    inputs["labels"] = correct_labels_tensor.to(device)
    
    return inputs

train_dataset_tokenized = train_dataset.map(lambda x: tokenize_function(x))
test_dataset_tokenized = test_dataset.map(lambda x: tokenize_function(x))

print(len(train_dataset_tokenized))
print(len(test_dataset_tokenized))

# Filter out examples that are too long
train_dataset_tokenized = train_dataset_tokenized.filter(lambda x: x['no_tokens'] <= 4000)
test_dataset_tokenized = test_dataset_tokenized.filter(lambda x: x['no_tokens'] <= 4000)

print(len(train_dataset_tokenized))
print(len(test_dataset_tokenized))

# Drop all columns except input_ids, attention_mask and labels
train_dataset_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


model = ModernBertForSequenceClassification.from_pretrained(
    model_name,
    token=os.environ["HF_TOKEN"],
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
    problem_type="multi_label_classification",
).to(device)

trainer = Trainer(
    model=model,
    train_dataset=train_dataset_tokenized,
    eval_dataset=test_dataset_tokenized,
    processing_class=tokenizer,
    args=TrainingArguments(
        gradient_accumulation_steps=1,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        dataloader_num_workers=64,
        num_train_epochs=1,
        learning_rate=1e-5,
        lr_scheduler_type="constant",
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=0.2,
        optim="adamw_torch_fused",
        report_to="wandb",
        output_dir=f"logs_{new_model_name}",
        save_steps=0.2,
        save_total_limit=1,
    ),
)

trainer.train()

# Save trained model
trainer.save_model(new_model_name)
tokenizer.save_pretrained(new_model_name)
