import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import os
import json
from utils import get_prompt_format, sorted_attributes
from tqdm import tqdm
import random
import copy
random.seed(42)

#ATTRIBUTES = ["focus"]
#ATTRIBUTES = ["local_or_sum"]
ATTRIBUTES = ["focus", "local_or_sum"]

TASK = "steerlm" # orso steerlm
#FILENAME = "EMNLP-ORSO/reshaped_datasets/new_fairytaleqa_train_orso_focus_local_or_sum.json"
FILENAME = "EMNLP-ORSO/reshaped_datasets/fairytaleqa_val.json"

dataset = json.load(open(FILENAME))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attributes_names = '_'.join(ATTRIBUTES)

model_name = f"llama32_1b_{TASK}_{attributes_names}_attribute"
#model_name = f"llama32_1b_after_clustering_{TASK}_{attributes_names}_attribute"

def get_prompt_message(example, attributes_list, wrong_attributes_list):
    messages = [
        {"role": "user", "content": get_prompt_format(example['content'], wrong_attributes_list)},
    ]
    return messages

def get_completion_message(example, attributes_list, wrong_attributes_list):
    messages = [
        {"role": "user", "content": get_prompt_format(example['content'], wrong_attributes_list)},
        {"role": "assistant", "content": f"Question: {example['question']}\nAnswer: {example['answer']}"},
    ]
    return messages

def calculate_loss_for_batch(batch):
    losses = []
    my_attributes_lists = []
    my_wrong_attributes_lists = []
    for example in batch:
        attributes_list = [(attr, example[attr]) for attr in ATTRIBUTES]
        my_attributes_lists.append(attributes_list)
        wrong_attributes_list = copy.deepcopy(attributes_list)
        rand_idx = random.randint(0, len(wrong_attributes_list) - 1)
        chosen_attr = wrong_attributes_list[rand_idx][0]
        wrong_attr_name = random.choice([a for a in sorted_attributes[chosen_attr] if a != wrong_attributes_list[rand_idx][1]])
        wrong_attributes_list[rand_idx] = (chosen_attr, wrong_attr_name)
        my_wrong_attributes_lists.append(wrong_attributes_list)


    messages_prompt = [get_prompt_message(example, my_attributes_lists[i], my_wrong_attributes_lists[i]) for i, example in enumerate(batch)]
    messages_completion = [get_completion_message(example, my_attributes_lists[i], my_wrong_attributes_lists[i]) for i, example in enumerate(batch)]
    
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
batch_size = 64

for i in tqdm(range(0, len(dataset), batch_size)):
    end_idx = min(i + batch_size, len(dataset))
    batch = dataset[i:end_idx]
    
    losses = calculate_loss_for_batch(batch)
    
    for example, loss in zip(batch, losses):
        example['loss'] = loss

# remove the .json extension from the filename

json.dump(dataset, open(f"{FILENAME[:-5]}_with_loss_negative_{TASK}_{attributes_names}.json", "w"), indent=4)
