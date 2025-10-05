import json
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def make_confusion_matrix(gold, predicted):
    conf_matrix = confusion_matrix(gold, predicted, labels=list(set(predicted + gold)))
    df_cm = pd.DataFrame(conf_matrix, index=list(set(predicted)), columns=list(set(predicted)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()

METHOD = 'steerlm' # orso steerlm
print(f"------------------------{METHOD}------------------------")

dataset = json.load(open(f"EACL-ORSO/final_classification/fairytaleqa_test_responses_llama32_1b_{METHOD}_focus_local_or_sum_attribute.json"))
#dataset = json.load(open(f"EACL-ORSO/final_classification/fairytaleqa_test_responses_llama32_1b_after_clustering_{METHOD}_focus_local_or_sum_attribute.json"))

"""
if METHOD == "steerlm":
    dataset = [d for d in dataset if d['logprobs'] < 4.8]
else:
    dataset = [d for d in dataset if d['logprobs'] < 8.5]
"""

predicted = [d['predicted_focus'] for d in dataset]
gold = [d['focus'] for d in dataset]

f1 = f1_score(gold, predicted, average='weighted')
accuracy = accuracy_score(gold, predicted)
print(f"ONLY FOCUS F1 Score: {f1:.3f}")
print(f"ONLY FOCUS Accuracy: {accuracy:.3f}")

predicted = [d['predicted_local_or_sum'] for d in dataset]
gold = [d['local_or_sum'] for d in dataset]
f1 = f1_score(gold, predicted, average='weighted')
accuracy = accuracy_score(gold, predicted)
print(f"ONLY LOCALSUM F1 Score: {f1:.3f}")
print(f"ONLY LOCALSUM Accuracy: {accuracy:.3f}")

predicted = [d['predicted_focus'] + d['predicted_local_or_sum'] for d in dataset]
gold = [d['focus'] + d['local_or_sum'] for d in dataset]

f1 = f1_score(gold, predicted, average='weighted')
accuracy = accuracy_score(gold, predicted)
print(f"BOTH F1 Score: {f1:.3f}")
print(f"BOTH Accuracy: {accuracy:.3f}")
