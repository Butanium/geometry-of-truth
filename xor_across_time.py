from probes import LRProbe
from utils import DataManager
import torch as th
import argparse
from probes import LRProbe
from utils import DataManager
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import pandas as pd
import time
from transformers import AutoConfig
from generate_acts import generate_acts

label_names = [
    "has_alice",
    "has_not",
    "label",
    "has_alice xor has_not",
    "has_alice xor label",
    "has_not xor label",
    "has_alice xor has_not xor label",
]

all_checkpoints = (
    [0] + [2**i for i in range(10)] + [1000 * 2**i for i in range(8)] + [143_000]
)


def xor_results(
    model,
    device,
    layers=None,
    seed=None,
    checkpoints=None,
    compute_acts=True,
    batch_size=1,
    chunk_size=25,
    random_init=False,
    shuffle=False,
    add_bos=False,
    noperiod=False,
):
    """
    Compute the accuracy of a logistic regression probe on some XOR features
    for a given model across depth and time.
    """
    if device == "auto":
        device = "cuda" if th.cuda.is_available() else "cpu"
        print(f"Using device {device}")
    if checkpoints == [-1]:
        checkpoints = all_checkpoints
    if checkpoints is None or checkpoints == []:
        checkpoints = ["trained"]
    if random_init:
        checkpoints.append("random init")
    if shuffle:
        checkpoints.append("shuffle")
    if layers is None:
        config = AutoConfig.from_pretrained(model)
        layers = list(range(config.num_hidden_layers + 1))
    if compute_acts:
        for checkpoint in checkpoints:
            if isinstance(checkpoint, int):
                revision = f"step{checkpoint}"
                checkpoint = f"Step {checkpoint}"
            else:
                revision = None
            print(f"Generating activations for checkpoint {checkpoint}")
            generate_acts(
                model,
                layers,
                ["cities_alice", "neg_cities_alice"],
                device=device,
                revision=revision,
                batch_size=batch_size,
                chunk_size=chunk_size,
                random_init=checkpoint == "random init",
                shuffle=checkpoint == "shuffle",
                add_bos=add_bos,
                noperiod=noperiod,
            )
    layer_accs = {}
    for layer in layers:
        checkpoint_accs = {}
        for checkpoint in checkpoints:
            if isinstance(checkpoint, int):
                revision = f"step{checkpoint}"
                checkpoint = f"Step {checkpoint}"
            else:
                revision = None
            print(f"Layer {layer}, {checkpoint}")
            accs = {}
            for label_name in label_names:
                dm = DataManager()
                for dataset in ["cities_alice", "neg_cities_alice"]:
                    dm.add_dataset(
                        dataset,
                        model,
                        layer,
                        seed=seed,
                        label=label_name,
                        center=False,
                        split=0.8,
                        device=device,
                        revision=revision,
                        shuffle=checkpoint == "shuffle",
                        random_init=checkpoint == "random init",
                    )
                acts, labels = dm.get("train")
                probe = LRProbe.from_data(acts, labels, bias=True, device=device)
                acts, labels = dm.get("val")
                acc = (probe(acts).round() == labels).float().mean()
                accs[label_name] = acc.item()
            checkpoint_accs[checkpoint] = accs
        layer_accs[layer] = checkpoint_accs
    return layer_accs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XOR Across Time")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument(
        "--device", type=str, help="Device (auto, cuda, cpu)", default="auto"
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        help="Layers to probe. Default: all",
        default=None,
    )
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        type=int,
        help="Checkpoints to probe. Default: -1 (all checkpoints on a log scale)."
        "If None are given, only the trained model is probed.",
        default=[-1],
    )
    parser.add_argument(
        "--no-acts",
        action="store_true",
        default=False,
        help="Set flag to disable computation of activations (if you already computed them). If it is not given,"
        "activations will be computed for all checkpoints before probing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for generating activations",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Number of activations to save per file",
    )
    parser.add_argument(
        "--random-init",
        action="store_true",
        default=False,
        help="Set flag to add a randomly initialized model in addition to the checkpoints",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=False,
        help="Set flag to add a model with the trained weights shuffled in addition to the checkpoints",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for experiments",
    )
    parser.add_argument(
        "--add-bos",
        action="store_true",
        default=False,
        help="Set flag to add the beginning of sentence token to the input sequences",
    )
    parser.add_argument(
        "--noperiod",
        action="store_true",
        default=False,
        help="Set flag to remove the period token from the input sequences",
    )
    args = parser.parse_args()

    # Get results
    all_accs = xor_results(
        args.model,
        args.device,
        layers=args.layers,
        checkpoints=args.checkpoints,
        compute_acts=not args.no_acts,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        random_init=args.random_init,
        shuffle=args.shuffle,
        seed=args.seed,
        add_bos=args.add_bos,
        noperiod=args.noperiod,
    )

    # Save results
    df = pd.DataFrame.from_dict(
        {
            (layer, revision): all_accs[layer][revision]
            for layer in all_accs
            for revision in all_accs[layer]
        },
        orient="index",
    )
    df.index.names = ["layer", "revision"]
    df.columns.name = "dataset"
    path = Path("results") / args.model / "xor_across_time"
    path.mkdir(parents=True, exist_ok=True)
    time_id = int(time.time())
    df.to_csv(path / f"results_{time_id}.csv")

    layers = list(all_accs.keys())
    checkpoint_nbs = list(all_accs[layers[0]].keys())
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "bar"}]])
    fig.update_layout(title=f"Accuracy across depth for {args.model}")

    for checkpoint_nb in checkpoint_nbs:
        for layer in layers:
            values = [all_accs[layer][checkpoint_nb][key] for key in label_names]
            fig.add_trace(
                go.Bar(
                    x=label_names,
                    y=values,
                    name=f"Layer {layer}, {checkpoint_nb}",
                )
            )

    # Add slider
    fig.update_layout(
        barmode="group",
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 16},
                    "visible": True,
                    "prefix": "Layer: ",
                    "suffix": "",
                },
                "pad": {"b": 10, "t": 50},
                "steps": [
                    {
                        "label": str(layer),
                        "method": "update",
                        "args": [{"visible": [layer == l for l in layers]}],
                    }
                    for layer in layers
                ],
            }
        ],
    )
    for data in fig.data:
        data.update(visible=f"Layer {layers[0]}" in data.name)
    fig.update_yaxes(range=[0, 1])
    fig.write_html(path / f"interactive_across_depth{time_id}.html")

    # Plot interactive results across time
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "bar"}]])

    for checkpoint_nb in checkpoint_nbs:
        for layer in layers:
            values = [all_accs[layer][checkpoint_nb][key] for key in label_names]
            fig.add_trace(
                go.Bar(
                    x=label_names,
                    y=values,
                    name=f"Layer {layer}, {checkpoint_nb}",
                )
            )
    # Add slider
    fig.update_layout(
        barmode="group",
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 16},
                    "visible": True,
                    "prefix": "Checkpoint: ",
                    "suffix": "",
                },
                "pad": {"b": 10, "t": 50},
                "steps": [
                    {
                        "label": str(checkpoint_nb),
                        "method": "update",
                        "args": [
                            {"visible": [checkpoint_nb == c for c in checkpoint_nbs]}
                        ],
                    }
                    for checkpoint_nb in checkpoint_nbs
                ],
            }
        ],
    )
    for data in fig.data:
        data.update(visible=f"Step {checkpoint_nbs[0]}" in data.name)
    fig.update_yaxes(range=[0, 1])
    # TODO: fix this before saving
    # fig.write_html(path / f"interactive_across_time{time_id}.html")
