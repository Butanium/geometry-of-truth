"""
add Alice/Bob to the start of statements in the given datasets
"""

import pandas as pd
import random

distractors = [
    "banana",
    "shed",
]
n_distractors = len(distractors)

dataset_names = [
    "cities",
    "neg_cities",
]

datasets = [
    pd.read_csv(f"datasets/{dataset_name}.csv") for dataset_name in dataset_names
]

n_statements = len(datasets[0])
assert all(len(dataset) == n_statements for dataset in datasets)

distractor_idxs = []
for idx in range(2 * n_distractors):
    distractor_idxs.extend([idx] * (n_statements // (2 * n_distractors)))
random.shuffle(distractor_idxs)


for dataset, name in zip(datasets, dataset_names):
    statements = []
    has_alices = []
    has_bananas = []
    has_xor = []
    for (_, row), distractor_idx in zip(dataset.iterrows(), distractor_idxs):
        alice_bob = "Alice" if distractor_idx % 2 == 0 else "Bob"
        distractor_real_idx = distractor_idx // 2

        statement = f"{alice_bob}: {row['statement']} {distractors[distractor_real_idx]}"
        statements.append(statement)
        has_alices.append(distractor_idx % 2 == 0)
        has_bananas.append(distractor_real_idx == 0)
        has_xor.append((distractor_idx % 2 == 0) ^ (distractor_real_idx == 0))
    dataset["statement"] = statements
    dataset["has_alice"] = has_alices
    dataset["has_banana"] = has_bananas
    dataset["has_alice_xor_has_banana"] = has_xor
    dataset.to_csv(f"datasets/{name}_alice_banana.csv", index=False)
