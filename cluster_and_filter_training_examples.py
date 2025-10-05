import json
from utils import sorted_attributes
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np

def cluster_texts(data, num_clusters=5000, model_name='all-MiniLM-L6-v2', use_minibatch=False):
    texts = [f"{item['question']}\n{item['answer']}" for item in data]

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)

    if use_minibatch:
        clusterer = MiniBatchKMeans(n_clusters=num_clusters, batch_size=100, random_state=42)
    else:
        clusterer = KMeans(n_clusters=num_clusters, random_state=42)

    labels = clusterer.fit_predict(embeddings)

    clusters = [[] for _ in range(num_clusters)]

    for label, item in zip(labels, data):
        clusters[label].append(item)

    for idx in range(num_clusters):
        clusters[idx] = list(sorted(clusters[idx], key=lambda x: x['loss']))

    return clusters

TASK = 'steerlm' # 'steerlm' or 'orso'

dataset = json.load(open(f"EMNLP-ORSO/reshaped_datasets/clean_new_fairytaleqa_train_{TASK}_focus_local_or_sum_with_loss_{TASK}_focus_local_or_sum.json"))
dataset_dict = {}

for attr1 in sorted_attributes['focus']:
    for attr2 in sorted_attributes['local_or_sum']:
        dataset_dict[(attr1, attr2)] = []

for example in dataset:
    if example['loss'] < 100:
        dataset_dict[(example['predicted_focus'], example['predicted_local_or_sum'])].append(example)

new_dataset = []
for (attr1, attr2), examples in dataset_dict.items():
    clusters = cluster_texts(examples)
    for cluster in clusters:
        if len(cluster) > 0:
            new_dataset.append(cluster[0])
        else:
            print(f"Empty cluster for attributes {attr1}, {attr2}")
        
json.dump(new_dataset, open(f"EMNLP-ORSO/bootstrapped_datasets/clustered_filtered_{TASK}.json", "w"), indent=4)
    