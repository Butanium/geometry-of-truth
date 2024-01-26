"""
    Add a constant column to dataset
"""

import pandas as pd
from pathlib import Path
import argparse
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


def add_constant(column_name, constant, dataset_names):
    ROOT = Path(__file__).parent.resolve()

    datasets = [
        pd.read_csv(ROOT / f"{dataset_name}.csv") for dataset_name in dataset_names
    ]
    new_datasets = []
    for dataset in datasets:
        n_rows = len(dataset["statement"])
        dataset[column_name] = [constant] * n_rows
        new_datasets.append(dataset)

    return new_datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a constant column to dataset")
    parser.add_argument("column_name", type=str, help="Name of the column to add")
    parser.add_argument("constant", type=bool, help="Value of the constant")
    parser.add_argument(
        "dataset_names", nargs="+", type=str, help="Names of the datasets"
    )

    args = parser.parse_args()

    datasets = add_constant(args.column_name, args.constant, args.dataset_names)

    for dataset, name in zip(datasets, args.dataset_names):
        dataset.to_csv(ROOT / f"{name}.csv", index=False)
