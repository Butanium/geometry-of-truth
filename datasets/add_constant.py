"""
    Add a constant column to dataset
"""

import pandas as pd

column_name = "has_not"
constant = False

dataset_names = [
    "companies_true_false"
]

datasets = [
    pd.read_csv(f"datasets/{dataset_name}.csv") for dataset_name in dataset_names
]

for dataset, name in zip(datasets, dataset_names):
    n_rows = len(dataset["statement"])

    dataset[column_name] = [constant] * n_rows
    dataset.to_csv(f"datasets/{name}_{column_name}.csv", index=False)
