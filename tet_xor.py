from probes import LRProbe
from utils import DataManager
import torch as th

label_names = [
    "has_alice",
    "has_en",
    "has_alice xor has_en",
]
DEVICE = "cpu"
if DEVICE == "auto":
    DEVICE = "cuda" if th.cuda.is_available() else "cpu"
model = "EleutherAI/pythia-70m-deduped"
all_accs = {}
for i in list(range(10)):
    revision = f"step{2**i}"
    accs = {}
    for label_name in label_names:
        dm = DataManager()
        dm.add_dataset(
            "sentences_alice",
            model,
            4,
            label=label_name,
            center=False,
            split=0.8,
            device=DEVICE,
            revision=revision,
        )
        acts, labels = dm.get("train")
        probe = LRProbe.from_data(acts, labels, bias=True, device=DEVICE)
        acts, labels = dm.get("val")
        acc = (probe(acts).round() == labels).float().mean()
        accs[label_name] = acc
    all_accs[revision] = accs

import plotly.express as px

all_accs_f = {k: {k2: v2.item() for k2, v2 in v.items()} for k, v in all_accs.items()}

fig = px.bar(all_accs_f, barmode='group')
fig.update_layout(xaxis_title="Feature", yaxis_title="Accuracy", legend_title="Revision")
fig.write_html("plot.html")