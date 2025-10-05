import json
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

test_dataset = json.load(open("EMNLP-ORSO/reshaped_datasets/fairytaleqa_test.json"))

METHOD = 'orso' # orso steerlm
print(f"------------------------{METHOD}------------------------")

dataset = json.load(open(f"EMNLP-ORSO/final_classification/fairytaleqa_test_responses_llama32_1b_{METHOD}_focus_local_or_sum_attribute.json"))

new_dataset = []
set_test = set()
for test_data in test_dataset:
    for data in dataset:
        if test_data['content'] == data['content'] and test_data['focus'] == data['focus'] and test_data['local_or_sum'] == data['local_or_sum'] and test_data['content'] + test_data['focus'] + test_data['local_or_sum'] not in set_test:
            new_dataset.append(data)
            set_test.add(test_data['content'] + test_data['focus'] + test_data['local_or_sum'])
            break

predicted = [d['predicted_focus'] for d in new_dataset]
gold = [d['focus'] for d in new_dataset]

f1 = f1_score(gold, predicted, average='weighted')
accuracy = accuracy_score(gold, predicted)
print(f"ONLY FOCUS F1 Score: {f1:.3f}")
print(f"ONLY FOCUS Accuracy: {accuracy:.3f}")

predicted = [d['predicted_local_or_sum'] for d in new_dataset]
gold = [d['local_or_sum'] for d in new_dataset]
f1 = f1_score(gold, predicted, average='weighted')
accuracy = accuracy_score(gold, predicted)
print(f"ONLY LOCALSUM F1 Score: {f1:.3f}")
print(f"ONLY LOCALSUM Accuracy: {accuracy:.3f}")

predicted = [d['predicted_focus'] + d['predicted_local_or_sum'] for d in new_dataset]
gold = [d['focus'] + d['local_or_sum'] for d in new_dataset]

f1 = f1_score(gold, predicted, average='weighted')
accuracy = accuracy_score(gold, predicted)
print(f"BOTH F1 Score: {f1:.3f}")
print(f"BOTH Accuracy: {accuracy:.3f}")
