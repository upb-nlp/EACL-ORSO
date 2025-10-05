import json

dataset = json.load(open("EACL-ORSO/new_fairytaleqa_train_responses_llama32_1b_orso_focus_local_or_sum_attribute.json"))
dataset = [data for data in dataset if data['predicted_focus'] != 'NO RESPONSE' and data['predicted_local_or_sum'] != 'NO RESPONSE']

json.dump(dataset, open("EACL-ORSO/reshaped_datasets/clean_new_fairytaleqa_train_orso_focus_local_or_sum.json", "w"), indent=4)


dataset = json.load(open("EACL-ORSO/new_fairytaleqa_train_responses_llama32_1b_steerlm_focus_local_or_sum_attribute.json"))
dataset = [data for data in dataset if data['predicted_focus'] != 'NO RESPONSE' and data['predicted_local_or_sum'] != 'NO RESPONSE']

json.dump(dataset, open("EACL-ORSO/reshaped_datasets/clean_new_fairytaleqa_train_steerlm_focus_local_or_sum.json", "w"), indent=4)
