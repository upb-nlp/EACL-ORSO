import torch
from datasets import Dataset
from trl import ORPOConfig
from orso_trainer import ORSOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
import random
random.seed(42)
import os
import json
from utils import get_prompt_format, sorted_attributes
import copy

#ATTRIBUTES = ["focus"]
#ATTRIBUTES = ["local_or_sum"]
ATTRIBUTES = ["focus", "local_or_sum"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attributes_names = '_'.join(ATTRIBUTES)
#model_name = f"llama32_1b_sft_{attributes_names}_attribute"
#new_model_name = f"llama32_1b_orso_{attributes_names}_attribute"
model_name = f"llama32_1b_orso_{attributes_names}_attribute"
new_model_name = f"llama32_1b_after_clustering_orso_{attributes_names}_attribute"

wandb.login(key=os.getenv('WANDB_TOKEN'))
wandb.init(
    project="ORSO",
    config={
        'task': f"FairytaleQA {attributes_names} Attribute Question Generation",
        'base_model': model_name,
        'model_name': new_model_name,
    }
)

def make_prompts(examples):
    attributes_list = [(attr, examples[attr]) for attr in ATTRIBUTES]
    messages_prompt_chosen = [
        {"role": "user", "content": get_prompt_format(examples['content'], attributes_list)},
    ]

    wrong_attributes_list = copy.deepcopy(attributes_list)
    rand_idx = random.randint(0, len(wrong_attributes_list) - 1)
    chosen_attr = wrong_attributes_list[rand_idx][0]
    wrong_attr_name = random.choice([a for a in sorted_attributes[chosen_attr] if a != wrong_attributes_list[rand_idx][1]])
    wrong_attributes_list[rand_idx] = (chosen_attr, wrong_attr_name)
    
    messages_prompt_rejected = [
        {"role": "user", "content": get_prompt_format(examples['content'], wrong_attributes_list)},
    ]

    text_prompt_chosen = tokenizer.apply_chat_template(messages_prompt_chosen, add_special_tokens=False, tokenize=False, add_generation_prompt=True)
    text_prompt_rejected = tokenizer.apply_chat_template(messages_prompt_rejected, add_special_tokens=False, tokenize=False, add_generation_prompt=True)

    text_response = f"Question: {examples['question']}\nAnswer: {examples['answer']}" + tokenizer.eos_token
    
    inputs = {
        'chosen_prompt': text_prompt_chosen,
        'rejected_prompt': text_prompt_rejected,
        'response': text_response,
    }
    return inputs

#data_train = json.load(open("EMNLP-ORSO/reshaped_datasets/new_fairytaleqa_train.json"))
data_train = json.load(open("EMNLP-ORSO/bootstrapped_datasets/clustered_filtered_orso_recompiled.json"))
data_val = json.load(open("EMNLP-ORSO/reshaped_datasets/fairytaleqa_val.json"))

random.shuffle(data_train)

# Convert to dataset
train_dataset = Dataset.from_list(data_train)
val_dataset = Dataset.from_list(data_val)

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token=os.getenv('HF_TOKEN'), trust_remote_code=True)
tokenizer.pad_token_id = 128002

train_dataset_prompts = train_dataset.map(lambda x: make_prompts(x))
val_dataset_prompts = val_dataset.map(lambda x: make_prompts(x))

print(len(train_dataset_prompts))
print(len(val_dataset_prompts))

# Drop all columns except input_ids, attention_mask and labels
train_dataset_prompts.set_format(type='torch', columns=['chosen_prompt', 'rejected_prompt', 'response'])
val_dataset_prompts.set_format(type='torch', columns=['chosen_prompt', 'rejected_prompt', 'response'])

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=os.getenv('HF_TOKEN'), trust_remote_code=True)

orso_trainer = ORSOTrainer(
    model,
    args=ORPOConfig(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=64,
        dataloader_num_workers=16,
        num_train_epochs=1,
        learning_rate=1e-5,
        lr_scheduler_type="constant",
        logging_steps=1,
        evaluation_strategy="steps",
        eval_steps=0.2,
        eval_on_start=True,
        optim="adamw_8bit",
        report_to="wandb",
        output_dir=f"logs_{new_model_name}",
        save_strategy="steps",
        save_steps=0.1,
        save_total_limit=1,
        max_length=3000,
        max_prompt_length=1000,
        beta=0.3,
        #fp16=True,
    ),
    train_dataset=train_dataset_prompts,
    eval_dataset=val_dataset_prompts,
    processing_class=tokenizer,
)

orso_trainer.train()
orso_trainer.save_model(new_model_name)
tokenizer.save_pretrained(new_model_name)