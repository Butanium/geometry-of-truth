"""
    Add a xor column to dataset
"""

import pandas as pd

column_name_1 = "has_alice"
column_name_2 = "has_not"

column_name = column_name_1 + " xor " + column_name_2

dataset_names = [
    "companies_true_false_has_not_alice"
]

datasets = [
    pd.read_csv(f"datasets/{dataset_name}.csv") for dataset_name in dataset_names
]

for dataset, name in zip(datasets, dataset_names):
    n_rows = len(dataset["statement"])

    dataset[column_name] = [x ^ y for x, y in zip(dataset[column_name_1], dataset[column_name_2])]
    dataset.to_csv(f"datasets/{name}_{column_name}.csv", index=False)
