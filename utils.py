import json
dataset = json.load(open("EMNLP-ORSO/reshaped_datasets/fairytaleqa_train.json"))
sorted_attributes = {
    'focus': list(sorted(list(set([data['focus'] for data in dataset])))),
    'local_or_sum': list(sorted(list(set([data['local_or_sum'] for data in dataset])))),
}

def get_explanation(attr_name, attr_value):
    if attr_name == 'focus':
        return f"The question must focus on {attr_value}."
    elif attr_name == 'local_or_sum':
        if attr_value == 'local':
            return "The question must be answerable based on a local context."
        elif attr_value == 'summary':
            return "The question must be answerable based on a wider context."
    
    raise ValueError(f"Unknown attribute name: {attr_name} and value: {attr_value}")

def get_prompt_format(context, attr_list):
    task_prompt = "Generate a question and an answer based on the following context."
    requirements_prompt = "The question must fulfill the following requirements:"
    explanations_prompt = "\n".join(["- " + get_explanation(attr_name, attr_value) for attr_name, attr_value in attr_list])
    return f"{task_prompt}\n\nContext:\n{context}\n\n{requirements_prompt}\n{explanations_prompt}"