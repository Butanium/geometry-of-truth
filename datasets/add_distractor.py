"""
add distractor word to the end of each statement in the given datasets
"""

import argparse
import pandas as pd
import random
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


def add_distractor(dataset_names, distractors):
    n_distractors = len(distractors)
    datasets = [
        pd.read_csv(f"datasets/{dataset_name}.csv") for dataset_name in dataset_names
    ]

    n_statements = len(datasets[0])
    assert all(len(dataset) == n_statements for dataset in datasets)

    distractor_idxs = []
    for idx in range(n_distractors):
        distractor_idxs.extend([idx] * (n_statements // n_distractors))
    if len(distractor_idxs) != n_statements:
        distractor_idxs.extend(
            random.choices(
                list(range(len(distractors))), k=n_statements - len(distractor_idxs)
            )
        )
    random.shuffle(distractor_idxs)
    new_datasets = []
    for dataset in datasets:
        dataset["distractor"] = [distractors[idx] for idx in distractor_idxs]
        dataset["statement"] = [
            statement + f" {distractor}"
            for statement, distractor in zip(
                dataset["statement"], dataset["distractor"]
            )
        ]
        for distractor in distractors:
            dataset[f"has_{distractor.lower()}"] = [
                (distractor == dataset["distractor"][idx]).astype(int)
                for idx in range(n_statements)
            ]
        new_datasets.append(dataset)
    return new_datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add distractor word to datasets")
    parser.add_argument("dataset_names", nargs="+", help="Names of the datasets")
    parser.add_argument(
        "--distractors",
        nargs="+",
        default=["banana", "shed"],
        help="Distractor words to add",
    )
    args = parser.parse_args()

    datasets = add_distractor(args.dataset_names, args.distractors)
    for dataset, name in zip(datasets, args.dataset_names):
        dataset.to_csv(ROOT / "{name}_distractors.csv", index=False)
