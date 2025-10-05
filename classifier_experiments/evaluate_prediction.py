import json
from sklearn.metrics import f1_score, accuracy_score

ATTRIBUTE = "local_or_sum" # focus local_or_sum

dataset = json.load(open(f"EACL-ORSO/classifier_experiments/fairytaleqa_test_predictions_{ATTRIBUTE}.json"))

predicted = [d[f"predicted_{ATTRIBUTE}"] for d in dataset]
gold = [d[ATTRIBUTE] for d in dataset]

# Calculate F1 score
f1 = f1_score(gold, predicted, average='weighted')
accuracy = accuracy_score(gold, predicted)

print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")

exit()
# Print a confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

conf_matrix = confusion_matrix(gold, predicted, labels=list(set(gold)))
df_cm = pd.DataFrame(conf_matrix, index=list(set(gold)), columns=list(set(predicted)))
plt.figure(figsize=(10, 8))
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()
