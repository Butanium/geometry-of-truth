from probes import LRProbe
from utils import DataManager
import torch as th
from tqdm import tqdm
import time
from transformers import AutoConfig
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="EleutherAI/pythia-410m-deduped")
parser.add_argument("--checkpoint-start", type=int, default=0)
args = parser.parse_args()
label_names = [
    "has_alice",
    "has_en",
    "has_alice xor has_en",
]
DEVICE = "cuda"
if DEVICE == "auto":
    DEVICE = "cuda" if th.cuda.is_available() else "cpu"
all_checkpoints = (
    [0] + [2**i for i in range(10)] + [1000 * 2**i for i in range(8)] + [143_000]
)
model = args.model
cfg = AutoConfig.from_pretrained(model)
all_accs = {}
for step in tqdm(all_checkpoints[args.checkpoint_start:]):
    revision = f"step{step}"
    accs = {}
    for label_name in label_names:
        dm = DataManager()
        dm.add_dataset(
            "sentences_alice",
            model,
            int(0.65 * cfg.num_hidden_layers),
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

fig = px.bar(all_accs_f, barmode="group")
fig.update_layout(
    xaxis_title="Feature",
    yaxis_title="Accuracy",
    legend_title="Revision",
    title=f"{model}, layer {int(0.65 * cfg.num_hidden_layers)}",
)
fig.write_html("plot" + str(time.time()) + ".html")
