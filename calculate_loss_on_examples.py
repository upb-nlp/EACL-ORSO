import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import os
import json
from utils import get_prompt_format
from tqdm import tqdm

#ATTRIBUTES = ["focus"]
#ATTRIBUTES = ["local_or_sum"]
ATTRIBUTES = ["focus", "local_or_sum"]

TASK = "steerlm" # orso steerlm
FILENAME = "EACL-ORSO/reshaped_datasets/clean_new_fairytaleqa_train_steerlm_focus_local_or_sum.json"

dataset = json.load(open(FILENAME))
start_index = 200000
end_index = min(start_index + 50000, len(dataset))
dataset = dataset[start_index:end_index]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attributes_names = '_'.join(ATTRIBUTES)

model_name = f"llama32_1b_{TASK}_{attributes_names}_attribute"

def get_prompt_message(example):
    attributes_list = [(attr, example[attr]) for attr in ATTRIBUTES]
    messages = [
        {"role": "user", "content": get_prompt_format(example['content'], attributes_list)},
    ]
    return messages

def get_completion_message(example):
    attributes_list = [(attr, example[attr]) for attr in ATTRIBUTES]
    messages = [
        {"role": "user", "content": get_prompt_format(example['content'], attributes_list)},
        {"role": "assistant", "content": f"Question: {example['question']}\nAnswer: {example['answer']}"},
    ]
    return messages

def calculate_loss_for_batch(batch):
    losses = []
    messages_prompt = [get_prompt_message(example) for example in batch]
    messages_completion = [get_completion_message(example) for example in batch]
    
    text_prompt = tokenizer.apply_chat_template(messages_prompt, add_special_tokens=False, tokenize=False, add_generation_prompt=True)
    text_completion = tokenizer.apply_chat_template(messages_completion, add_special_tokens=False, tokenize=False)
    
    inputs_prompt = tokenizer(text_prompt, return_tensors="pt", padding=True, max_length=2*2048, truncation=True)
    inputs_completion = tokenizer(text_completion, return_tensors="pt", padding=True, max_length=2*2048, truncation=True)
    
    # Calculate the loss
    inputs_prompt = inputs_prompt.to(device)
    inputs_completion = inputs_completion.to(device)

    with torch.no_grad():
        outputs = model(input_ids=inputs_completion['input_ids'], attention_mask=inputs_completion['attention_mask'])
        logits = outputs.logits

    for logit, input, whole in zip(logits, inputs_prompt['input_ids'], inputs_completion['input_ids']):
        # Remove padding
        padding = torch.count_nonzero(whole == tokenizer.pad_token_id)
        whole = whole[padding:]
        padding = torch.count_nonzero(input == tokenizer.pad_token_id)
        input = input[padding:]

        logit = logit[:-1]
        good_logit = logit[-(len(whole) - len(input)):]
        good_label = whole[len(input):]

        loss = loss_fn(
            good_logit,
            good_label,
        )
        losses.append(loss.item())
    return losses



tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token=os.getenv('HF_TOKEN'), trust_remote_code=True)
tokenizer.pad_token_id = 128002

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=os.getenv('HF_TOKEN'), trust_remote_code=True)

loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
batch_size = 16

for i in tqdm(range(0, len(dataset), batch_size)):
    end_idx = min(i + batch_size, len(dataset))
    batch = dataset[i:end_idx]
    
    losses = calculate_loss_for_batch(batch)
    
    for example, loss in zip(batch, losses):
        example['loss'] = loss

# remove the .json extension from the filename

json.dump(dataset, open(f"{FILENAME[:-5]}_{start_index}_{end_index}_with_loss_{TASK}_{attributes_names}.json", "w"), indent=4)
