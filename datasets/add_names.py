"""
add Alice/Bob to the start of statements in the given datasets
"""

import pandas as pd
import random
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


def add_names_to_statements(dataset_names, person_names):
    datasets = [
        pd.read_csv(f"datasets/{dataset_name}.csv") for dataset_name in dataset_names
    ]

    n_statements = len(datasets[0])
    assert all(len(dataset) == n_statements for dataset in datasets)

    distractor_idxs = []
    for idx in range(len(person_names)):
        distractor_idxs.extend([idx] * (n_statements // len(person_names)))
    if len(distractor_idxs) != n_statements:
        distractor_idxs.extend(
            random.choices(
                list(range(len(person_names))), k=n_statements - len(distractor_idxs)
            )
        )
    random.shuffle(distractor_idxs)
    new_datasets = []
    for dataset in datasets:
        statements = []
        has_persons = [[] for _ in person_names]
        for (_, row), distractor_idx in zip(dataset.iterrows(), distractor_idxs):
            statement = f"{person_names[distractor_idx]}: {row['statement']}"
            statements.append(statement)
            for idx in range(len(person_names)):
                if idx != distractor_idx:
                    has_persons[idx].append(0)
                else:
                    has_persons[idx].append(1)
        dataset["statement"] = statements
        for idx, person_name in enumerate(person_names):
            dataset[f"has_{person_name.lower()}"] = has_persons[idx]
        new_datasets.append(dataset)
    return new_datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add Alice/Bob to statements in datasets"
    )
    parser.add_argument("dataset_names", nargs="+", type=str, help="Dataset names")
    parser.add_argument(
        "names", nargs="+", type=str, help="Names", default=["Alice", "Bob"]
    )
    args = parser.parse_args()
    assert len(args.names) >= 2, "Must provide at least two names"

    datasets = add_names_to_statements(args.dataset_names, args.names)

    for dataset, name in zip(datasets, args.dataset_names):
        dataset.to_csv(ROOT / f"{name}_names.csv", index=False)
