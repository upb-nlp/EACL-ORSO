import json
import plotly.graph_objects as go

test_dataset = json.load(open("EACL-ORSO/reshaped_datasets/fairytaleqa_val.json"))

METHOD = 'orso' # orso steerlm
print(f"------------------------{METHOD}------------------------")

dataset = json.load(open(f"EACL-ORSO/fairytaleqa_val_responses_llama32_1b_after_clustering_{METHOD}_focus_local_or_sum_attribute.json"))

sure_dataset = []
set_test = set()
for test_data in test_dataset:
    for data in dataset:
        if test_data['content'] == data['content'] and test_data['focus'] == data['focus'] and test_data['local_or_sum'] == data['local_or_sum'] and test_data['content'] + test_data['focus'] + test_data['local_or_sum'] not in set_test:
            sure_dataset.append(data)
            set_test.add(test_data['content'] + test_data['focus'] + test_data['local_or_sum'])
            break

unsure_dataset = []
for data in dataset:
    found = False
    for sd in sure_dataset:
        if data['content'] == sd['content'] and data['focus'] == sd['focus'] and data['local_or_sum'] == sd['local_or_sum']:
            found = True
            break
    if not found:
        unsure_dataset.append(data)

sure_dataset = [d['logprobs'] for d in sure_dataset]
unsure_dataset = [d['logprobs'] for d in unsure_dataset]

# Calculate the mean, std Q1, q2, Q3 for sure
mean_sure = sum(sure_dataset) / len(sure_dataset)
std_sure = (sum((x - mean_sure) ** 2 for x in sure_dataset) / len(sure_dataset)) ** 0.5
q1_sure = sorted(sure_dataset)[int(len(sure_dataset) * 0.25)]
q2_sure = sorted(sure_dataset)[int(len(sure_dataset) * 0.5)]
q3_sure = sorted(sure_dataset)[int(len(sure_dataset) * 0.75)]
qx_sure = sorted(sure_dataset)[int(len(sure_dataset) * 0.9)]
print(f"Sure Dataset - Mean: {mean_sure:.3f}, Std: {std_sure:.3f}, Q1: {q1_sure:.3f}, Q2: {q2_sure:.3f}, Q3: {q3_sure:.3f} QX: {qx_sure:.3f}")


fig = go.Figure()
# make them normalized as a probability distribution
fig.add_trace(go.Histogram(
    x=sure_dataset,
    name=f"{METHOD.capitalize()} - Sure",
    opacity=0.75,
    histnorm='probability density',
))
fig.add_trace(go.Histogram(
    x=unsure_dataset,
    name=f"{METHOD.capitalize()} - Unsure",
    opacity=0.75,
    histnorm='probability density',
))
fig.update_layout(
    title=f'Histogram of {METHOD.capitalize()} Sure vs Unsure',
    xaxis_title='Negative Log Likelihood',
    barmode='overlay',
    xaxis=dict(
        tickmode='auto'  # or use tickmode='linear', dtick=0.2
    ),
)
fig.write_image(f"EACL-ORSO/{METHOD}_Val_Sure_vs_Unsure_after_clustering.png")
