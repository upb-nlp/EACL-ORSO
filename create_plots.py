import json
import plotly.graph_objects as go

MOMENT = '_after_clustering' # _after_clustering

data_orso_chosen = json.load(open(f"EMNLP-ORSO/reshaped_datasets/fairytaleqa_val_with_loss{MOMENT}_orso_focus_local_or_sum.json"))
data_orso_chosen = [d['loss'] for d in data_orso_chosen]

data_orso_rejected = json.load(open(f"EMNLP-ORSO/reshaped_datasets/fairytaleqa_val_with_loss_negative{MOMENT}_orso_focus_local_or_sum.json"))
data_orso_rejected = [d['loss'] for d in data_orso_rejected]

data_steerlm_chosen = json.load(open(f"EMNLP-ORSO/reshaped_datasets/fairytaleqa_val_with_loss{MOMENT}_steerlm_focus_local_or_sum.json"))
data_steerlm_chosen = [d['loss'] for d in data_steerlm_chosen]
data_steerlm_rejected = json.load(open(f"EMNLP-ORSO/reshaped_datasets/fairytaleqa_val_with_loss_negative{MOMENT}_steerlm_focus_local_or_sum.json"))
data_steerlm_rejected = [d['loss'] for d in data_steerlm_rejected]

# Plot data_orso_chosen and data_orso_rejected on the same plot as histograms
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=data_orso_chosen,
    name='Correct',
    opacity=0.75,
    histnorm='probability density',
))
fig.add_trace(go.Histogram(
    x=data_orso_rejected,
    name='Incorrect',
    opacity=0.75,
    histnorm='probability density',
))
fig.update_layout(
    font=dict(size=30),
    width=1600,
    height=1200,
    margin=dict(l=5, r=5, t=5, b=5),
    #title='Histogram of ORSO Correct vs. Incorrect',
    xaxis_title='Negative Log Likelihood',
    yaxis_title='Probability Density',
    barmode='overlay',
    xaxis_title_font=dict(size=30),
    yaxis_title_font=dict(size=30),
    xaxis=dict(
        tickmode='auto',  # or use tickmode='linear', dtick=0.2
        tickfont=dict(size=30),
    ),
    yaxis=dict(
        range=[0, 0.03],
    ),
)
fig.write_image(f"EMNLP-ORSO/orso_chosen_vs_rejected{MOMENT}.png")

fig = go.Figure()
fig.add_trace(go.Histogram(
    x=data_steerlm_chosen,
    name='Correct',
    opacity=0.75,
    histnorm='probability density',
))
fig.add_trace(go.Histogram(
    x=data_steerlm_rejected,
    name='Incorrect',
    opacity=0.75,
    histnorm='probability density',
))
fig.update_layout(
    font=dict(size=30),
    width=1600,
    height=1200,
    margin=dict(l=5, r=5, t=5, b=5),
    #title='Histogram of SteerLM Correct vs. Incorrect',
    xaxis_title='Negative Log Likelihood',
    yaxis_title='Probability Density',
    barmode='overlay',
    xaxis_title_font=dict(size=30),
    yaxis_title_font=dict(size=30),
    xaxis=dict(
        tickmode='auto',
        tickfont=dict(size=30),
    ),
    yaxis=dict(
        range=[0, 0.03],
    ),
)
fig.write_image(f"EMNLP-ORSO/steerlm_chosen_vs_rejected{MOMENT}.png")