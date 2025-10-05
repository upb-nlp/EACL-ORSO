import json
from utils import sorted_attributes
initial_dataset = json.load(open("EMNLP-ORSO/reshaped_datasets/fairytaleqa_train.json"))

new_dataset_steerlm = json.load(open("EMNLP-ORSO/new_fairytaleqa_train_responses_llama32_1b_steerlm_focus_local_or_sum_attribute.json"))
new_dataset_orso = json.load(open("EMNLP-ORSO/new_fairytaleqa_train_responses_llama32_1b_orso_focus_local_or_sum_attribute.json"))

new_dataset_orso = [data for data in new_dataset_orso if data['focus'] == data['predicted_focus'] and data['local_or_sum'] == data['predicted_local_or_sum']]
new_dataset_steerlm = [data for data in new_dataset_steerlm if data['focus'] == data['predicted_focus'] and data['local_or_sum'] == data['predicted_local_or_sum']]

counter_focus_initial = {k: 0 for k in sorted_attributes['focus']}
counter_focus_initial['NO RESPONSE'] = 0
counter_local_or_sum_initial = {k: 0 for k in sorted_attributes['local_or_sum']}
counter_local_or_sum_initial['NO RESPONSE'] = 0
for data in initial_dataset:
    counter_focus_initial[data['focus']] += 1
    counter_local_or_sum_initial[data['local_or_sum']] += 1

counter_focus_initial = {k: v / len(initial_dataset) for k, v in counter_focus_initial.items()}
counter_local_or_sum_initial = {k: v / len(initial_dataset) for k, v in counter_local_or_sum_initial.items()}

counter_focus_steerlm = {k: 0 for k in sorted_attributes['focus']}
counter_focus_steerlm['NO RESPONSE'] = 0
counter_local_or_sum_steerlm = {k: 0 for k in sorted_attributes['local_or_sum']}
counter_local_or_sum_steerlm['NO RESPONSE'] = 0
for data in new_dataset_steerlm:
    counter_focus_steerlm[data['predicted_focus']] += 1
    counter_local_or_sum_steerlm[data['predicted_local_or_sum']] += 1
counter_focus_steerlm = {k: v / len(new_dataset_steerlm) for k, v in counter_focus_steerlm.items()}
counter_local_or_sum_steerlm = {k: v / len(new_dataset_steerlm) for k, v in counter_local_or_sum_steerlm.items()}

counter_focus_orso = {k: 0 for k in sorted_attributes['focus']}
counter_focus_orso['NO RESPONSE'] = 0
counter_local_or_sum_orso = {k: 0 for k in sorted_attributes['local_or_sum']}
counter_local_or_sum_orso['NO RESPONSE'] = 0
for data in new_dataset_orso:
    counter_focus_orso[data['predicted_focus']] += 1
    counter_local_or_sum_orso[data['predicted_local_or_sum']] += 1
counter_focus_orso = {k: v / len(new_dataset_orso) for k, v in counter_focus_orso.items()}
counter_local_or_sum_orso = {k: v / len(new_dataset_orso) for k, v in counter_local_or_sum_orso.items()}


# counter_focus_initial and counter_focus_steerlm and counter_focus_orso dicts have the same keys. Print them in a human readable table, with 2 decimals
print(f"{'Focus':<20} {'Initial':<10} {'SteerLM':<10} {'Orso':<10}")
for k in sorted_attributes['focus'] + ['NO RESPONSE']:
    print(f"{k:<20} {counter_focus_initial[k]:<10.2f} {counter_focus_steerlm[k]:<10.2f} {counter_focus_orso[k]:<10.2f}")

print()
print()
print(f"{'Local or Sum':<20} {'Initial':<10} {'SteerLM':<10} {'Orso':<10}")
for k in sorted_attributes['local_or_sum'] + ['NO RESPONSE']:
    print(f"{k:<20} {counter_local_or_sum_initial[k]:<10.2f} {counter_local_or_sum_steerlm[k]:<10.2f} {counter_local_or_sum_orso[k]:<10.2f}")
# Print the number of examples in each dataset
print(f"Initial dataset size: {len(initial_dataset)}")
print(f"SteerLM dataset size: {len(new_dataset_steerlm)}")
print(f"Orso dataset size: {len(new_dataset_orso)}")

