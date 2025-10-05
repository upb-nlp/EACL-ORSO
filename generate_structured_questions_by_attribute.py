import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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
TESTING = 'orso' # 'steerlm' or 'orso'

attributes_names = '_'.join(ATTRIBUTES)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

access_token = os.getenv('HF_TOKEN')

model_name = f"llama32_1b_{TESTING}_{attributes_names}_attribute"

guided_decoding_params_regex = GuidedDecodingParams(regex=r"Question:[^?]+\?\nAnswer:.*")
llm = LLM(model=model_name, dtype=torch.bfloat16, max_model_len=2048*2, enable_prefix_caching=True)
sampling_params = SamplingParams(
    guided_decoding=guided_decoding_params_regex,
    max_tokens=256,
    temperature=1,
    top_p=0.9,
    min_p=0.2,
    logprobs=1,
    n=5,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token=access_token)
tokenizer.pad_token_id = 128002

dataset_texts = json.load(open("EMNLP-ORSO/reshaped_datasets/fairytaleqa_val.json"))
dataset_texts = [data['content'] for data in dataset_texts]
dataset_texts = list(sorted(list(set(dataset_texts))))

prompts = []
for text in dataset_texts:
    if len(ATTRIBUTES) == 1:
        # For each attribute value, generate a question and an answer
        for attribute in sorted_attributes[ATTRIBUTES[0]]:
            prompts.append(get_prompt_format(text, [(ATTRIBUTES[0], attribute)]))
    else:
        # For each attribute value, generate a question and an answer
        for attribute in sorted_attributes[ATTRIBUTES[0]]:
            for attribute2 in sorted_attributes[ATTRIBUTES[1]]:
                prompts.append(get_prompt_format(text, [(ATTRIBUTES[0], attribute), (ATTRIBUTES[1], attribute2)]))

messages_list = [[{'role': 'user', 'content': prompt}] for prompt in prompts]
#chat_template_prompts = [tokenizer.apply_chat_template(messages, add_special_tokens=False, tokenize=False, add_generation_prompt=True) for messages in messages_list]

responses = []
responses_logprobs = []
batch_size = 64
for i in tqdm(range(0, len(messages_list), batch_size)):
    end_interval = min(i+batch_size, len(messages_list))

    texts = messages_list[i:end_interval]

    completion = llm.chat(texts, sampling_params, use_tqdm=False)

    res = []
    rlgp = []
    for comp in completion:
        outs = comp.outputs
        list_res = []
        for o in outs:
            list_res.append({
                'text': o.text,
                'cumulative_logprob': -o.cumulative_logprob,
            })
        list_res = sorted(list_res, key=lambda x: x['cumulative_logprob'])
        res.append(list_res[0]['text'])
        rlgp.append(list_res[0]['cumulative_logprob'])

    responses += res
    responses_logprobs += rlgp

dict_responses = []
i = -1
for text in dataset_texts:
    if len(ATTRIBUTES) == 1:
        attributes_set = sorted_attributes[ATTRIBUTES[0]]
        for attribute in attributes_set:
            i += 1
            res = responses[i]
            try:
                question = res.split("Answer:")[0].replace("Question:", "").strip()
                answer = res.split("Answer:")[1].strip()
            except:
                question = None
                answer = None

            dict_responses.append({
                "content": text,
                ATTRIBUTES[0]: attribute,
                "question": question,
                "answer": answer,
                'whole_response': res,
                'logprobs': responses_logprobs[i],
            })
    else:
        attributes_set = sorted_attributes[ATTRIBUTES[0]]
        for attribute in attributes_set:
            attributes_set2 = sorted_attributes[ATTRIBUTES[1]]
            for attribute2 in attributes_set2:
                i += 1
                res = responses[i]
                try:
                    question = res.split("Answer:")[0].replace("Question:", "").strip()
                    answer = res.split("Answer:")[1].strip()
                except:
                    question = None
                    answer = None

                dict_responses.append({
                    "content": text,
                    ATTRIBUTES[0]: attribute,
                    ATTRIBUTES[1]: attribute2,
                    "question": question,
                    "answer": answer,
                    'whole_response': res,
                    'logprobs': responses_logprobs[i],
                })

json.dump(dict_responses, open(f"EMNLP-ORSO/fairytaleqa_val_responses_{model_name.split('/')[-1]}.json", "w"), indent=4)