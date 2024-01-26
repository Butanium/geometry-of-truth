"""
    Add a xor column to datasets
"""

import argparse
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


def add_xor_column(dataset_names, column_names):
    xor_column_name = " xor ".join(column_names)

    datasets = [
        pd.read_csv(f"datasets/{dataset_name}.csv") for dataset_name in dataset_names
    ]

    result_datasets = []

    for dataset in datasets:
        xor_result = dataset[column_names[0]]
        for column_name in column_names[1:]:
            xor_result = xor_result ^ dataset[column_name]

        dataset[xor_column_name] = xor_result.astype(int)
        result_datasets.append(dataset)

    return result_datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add xor column to datasets")
    parser.add_argument("dataset_names", nargs="+", help="Names of the datasets")
    parser.add_argument("column_names", nargs="+", help="Names of the columns")
    args = parser.parse_args()

    result_datasets = add_xor_column(args.dataset_names, args.column_names)

    for dataset, name in zip(result_datasets, args.dataset_names):
        dataset.to_csv(ROOT / f"{name}.csv", index=False)
