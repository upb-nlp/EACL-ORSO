import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import wandb
import random
from trl import DataCollatorForCompletionOnlyLM
import os
import json
from utils import get_prompt_format

#ATTRIBUTES = ["focus"]
#ATTRIBUTES = ["local_or_sum"]
ATTRIBUTES = ["focus", "local_or_sum"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attributes_names = '_'.join(ATTRIBUTES)
model_name = f"meta-llama/Llama-3.2-1B-Instruct"
new_model_name = f"llama32_1b_sft_{attributes_names}_attribute"

wandb.login(key=os.getenv('WANDB_TOKEN'))
wandb.init(
    project="ORSO",
    config={
        'task': f"FairytaleQA {attributes_names} Attribute Question Generation",
        'base_model': model_name,
        'model_name': new_model_name,
    }
)

def tokenize_function(examples):
    attributes_list = [(attr, examples[attr]) for attr in ATTRIBUTES]
    messages = [
        {"role": "user", "content": get_prompt_format(examples['content'], attributes_list)},
        {"role": "system", "content": f"Question: {examples['question']}\nAnswer: {examples['answer']}"},
    ]

    text_example = tokenizer.apply_chat_template(messages, add_special_tokens=False, tokenize=False)

    inputs = tokenizer(text_example, return_tensors="pt", max_length=2*2048, truncation=True)
    inputs["no_tokens"] = inputs["input_ids"].shape[1] + 1
    inputs["attention_mask"] = inputs["attention_mask"].squeeze()
    inputs["input_ids"] = inputs["input_ids"].squeeze()

    return inputs


data_train = json.load(open("EACL-ORSO/reshaped_datasets/fairytaleqa_train.json"))
data_val = json.load(open("EACL-ORSO/reshaped_datasets/fairytaleqa_val.json"))

random.shuffle(data_train)

# Convert to dataset
train_dataset = Dataset.from_list(data_train)
val_dataset = Dataset.from_list(data_val)

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token=os.getenv('HF_TOKEN'), trust_remote_code=True)
tokenizer.pad_token_id = 128002


train_dataset_tokenized = train_dataset.map(lambda x: tokenize_function(x))
val_dataset_tokenized = val_dataset.map(lambda x: tokenize_function(x))

print(len(train_dataset_tokenized))
print(len(val_dataset_tokenized))

# Filter out examples that are too long
train_dataset_tokenized = train_dataset_tokenized.filter(lambda x: x['no_tokens'] <= 4000)
val_dataset_tokenized = val_dataset_tokenized.filter(lambda x: x['no_tokens'] <= 4000)

print(len(train_dataset_tokenized))
print(len(val_dataset_tokenized))

# Drop all columns except input_ids, attention_mask and labels
train_dataset_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
val_dataset_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=os.getenv('HF_TOKEN'), trust_remote_code=True)

trainer = Trainer(
    model=model,
    train_dataset=train_dataset_tokenized,
    eval_dataset=val_dataset_tokenized,
    tokenizer=tokenizer,
    args=TrainingArguments(
        gradient_accumulation_steps=64,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        dataloader_num_workers=16,
        num_train_epochs=1,
        learning_rate=5e-6,
        lr_scheduler_type="constant",
        logging_steps=1,
        evaluation_strategy="steps",
        eval_steps=0.2,
        eval_on_start=True,
        optim="adamw_8bit",
        report_to="wandb",
        output_dir=f"logs_{new_model_name}",
        save_steps=0.2,
        save_total_limit=1,
    ),
    data_collator=DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, mlm=False, return_tensors="pt", response_template="<|start_header_id|>system<|end_header_id|>")
)

trainer.train()

# Save trained model
trainer.save_model(new_model_name)
tokenizer.save_pretrained(new_model_name)
