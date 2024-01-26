"""
add Alice/Bob to the start of statements in the given datasets
"""

import pandas as pd
from pathlib import Path
from add_names import add_names_to_statements
from add_distractor import add_distractor
from add_xor import add_xor_column
import argparse

ROOT = Path(__file__).parent.resolve()


def main():
    parser = argparse.ArgumentParser(
        description="Add Alice/Bob to statements in datasets"
    )
    parser.add_argument(
        "--distractors",
        nargs="+",
        default=["banana", "shed"],
        help="List of distractors",
    )
    parser.add_argument(
        "--names", nargs="+", default=["Alice", "Bob"], help="List of names"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cities", "neg_cities", "companies_true_false"],
        help="List of dataset names",
    )
    args = parser.parse_args()

    distractors = args.distractors
    names = args.names
    dataset_names = args.datasets

    datasets = add_names_to_statements(dataset_names, names)
    datasets = add_distractor(dataset_names, distractors)
    for name, distractor in zip(names, distractors):
        datasets = add_xor_column(
            dataset_names, [f"has_{name.lower()}", f"has_{distractor.lower()}"]
        )

    for dataset, name in zip(datasets, dataset_names):
        dataset.to_csv(ROOT / f"{name}_names_distractors.csv", index=False)


if __name__ == "__main__":
    main()
