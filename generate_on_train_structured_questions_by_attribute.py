import torch
from tqdm import tqdm 
import json
import random
random.seed(42)
import os
from utils import sorted_attributes, get_prompt_format
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

#ATTRIBUTES = ["focus"]
#ATTRIBUTES = ["local_or_sum"]
ATTRIBUTES = ["focus", "local_or_sum"]
TESTING = 'steerlm' # 'steerlm' or 'orso'

attributes_names = '_'.join(ATTRIBUTES)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

access_token = os.getenv('HF_TOKEN')

model_name = f"llama32_1b_{TESTING}_{attributes_names}_attribute"

guided_decoding_params_regex = GuidedDecodingParams(regex=r"Question:[^?]+\?\nAnswer:.*")
llm = LLM(model=model_name, dtype=torch.bfloat16, max_model_len=2048*2, enable_prefix_caching=True, gpu_memory_utilization=0.9)
sampling_params = SamplingParams(
    n=5,
    guided_decoding=guided_decoding_params_regex,
    max_tokens=256,
    temperature=1,
    top_p=0.9,
    min_p=0.2,
)
dataset_texts = json.load(open("EACL-ORSO/reshaped_datasets/fairytaleqa_train.json"))
dataset_texts = [data['content'] for data in dataset_texts]
dataset_texts = list(sorted(list(set(dataset_texts))))

prompts = []
new_dataset = []
for text in dataset_texts:
    if len(ATTRIBUTES) == 1:
        # For each attribute value, generate a question and an answer
        for attribute in sorted_attributes[ATTRIBUTES[0]]:
            prompts.append(get_prompt_format(text, [(ATTRIBUTES[0], attribute)]))
            new_dataset.append({
                'content': text,
                ATTRIBUTES[0]: attribute,
            })
    else:
        # For each attribute value, generate a question and an answer
        for attribute in sorted_attributes[ATTRIBUTES[0]]:
            for attribute2 in sorted_attributes[ATTRIBUTES[1]]:
                prompts.append(get_prompt_format(text, [(ATTRIBUTES[0], attribute), (ATTRIBUTES[1], attribute2)]))
                new_dataset.append({
                    'content': text,
                    ATTRIBUTES[0]: attribute,
                    ATTRIBUTES[1]: attribute2,
                })

messages_list = [[{'role': 'user', 'content': prompt}] for prompt in prompts]
#chat_template_prompts = [tokenizer.apply_chat_template(messages, add_special_tokens=False, tokenize=False, add_generation_prompt=True) for messages in messages_list]

responses = []
batch_size = 64
for i in tqdm(range(0, len(messages_list), batch_size)):
    end_interval = min(i+batch_size, len(messages_list))

    texts = messages_list[i:end_interval]

    completion = llm.chat(texts, sampling_params, use_tqdm=False)

    #res = [r.outputs[0].text for r in completion]
    res = []
    
    for r, idx in zip(completion, range(i, end_interval)):
        new_dataset[idx]['all_generations'] = [{'whole_response': o.text} for o in r.outputs]
        for gen in new_dataset[idx]['all_generations']:
            try:
                question = gen['whole_response'].split("Answer:")[0].replace("Question:", "").strip()
                answer = gen['whole_response'].split("Answer:")[1].strip()
            except:
                question = None
                answer = None
            gen['question'] = question
            gen['answer'] = answer
            del gen['whole_response']
    del completion
    

json.dump(new_dataset, open(f"EACL-ORSO/new_fairytaleqa_train_responses_{model_name.split('/')[-1]}.json", "w"), indent=4)
