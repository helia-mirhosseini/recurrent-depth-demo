import argparse
import csv
import json
import random
from pathlib import Path
from typing import List, Tuple, Set


# ---------------------------------------------------------
# Balanced sequence generation
# ---------------------------------------------------------

def generate_balanced_sequence(length: int, rng: random.Random) -> str:
    """
    Generate a valid balanced-parentheses sequence.

    Example:
        (()())
        ((())())

    A balanced sequence must:
        1. contain the same number of '(' and ')'
        2. never have a negative running balance
        3. finish with balance == 0

    length must be even.
    """
    if length % 2 != 0:
        raise ValueError("Balanced-parentheses sequences must have even length.")

    num_pairs = length // 2

    sequence = []
    opens_used = 0
    closes_used = 0
    balance = 0

    for _ in range(length):
        can_open = opens_used < num_pairs
        can_close = closes_used < num_pairs and balance > 0

        if can_open and can_close:
            # Randomly choose either action.
            if rng.random() < 0.5:
                sequence.append("(")
                opens_used += 1
                balance += 1
            else:
                sequence.append(")")
                closes_used += 1
                balance -= 1

        elif can_open:
            sequence.append("(")
            opens_used += 1
            balance += 1

        elif can_close:
            sequence.append(")")
            closes_used += 1
            balance -= 1

        else:
            raise RuntimeError("Invalid generation state.")

    result = "".join(sequence)

    assert is_balanced(result)

    return result


# ---------------------------------------------------------
# Validation utility
# ---------------------------------------------------------

def is_balanced(sequence: str) -> bool:
    """
    Check whether a parentheses sequence is balanced.
    """
    balance = 0

    for char in sequence:
        if char == "(":
            balance += 1
        elif char == ")":
            balance -= 1
        else:
            raise ValueError(f"Unexpected character: {char}")

        # A closing parenthesis appeared before
        # a corresponding opening parenthesis.
        if balance < 0:
            return False

    return balance == 0


# ---------------------------------------------------------
# Hard negative generation
# ---------------------------------------------------------

def generate_equal_count_unbalanced(
    length: int,
    rng: random.Random,
) -> str:
    """
    Generate an UNBALANCED sequence while preserving:

        number of '(' == number of ')'

    Therefore the model cannot solve these examples simply
    by counting opening and closing parentheses.

    Example:

        ())(()

    Counts:
        '(' = 3
        ')' = 3

    But the sequence becomes invalid because its prefix
    balance goes negative.
    """
    if length % 2 != 0:
        raise ValueError(
            "Equal-count unbalanced sequences require even length."
        )

    num_pairs = length // 2

    while True:
        chars = ["("] * num_pairs + [")"] * num_pairs
        rng.shuffle(chars)

        sequence = "".join(chars)

        if not is_balanced(sequence):
            return sequence


def generate_count_mismatch_unbalanced(
    length: int,
    rng: random.Random,
) -> str:
    """
    Generate an obviously unbalanced sequence whose total
    numbers of '(' and ')' differ.

    These examples are still useful, but they should not
    dominate the negative class because otherwise the model
    may learn the easy counting shortcut.
    """

    while True:
        sequence = "".join(
            rng.choice(["(", ")"])
            for _ in range(length)
        )

        if not is_balanced(sequence):
            if sequence.count("(") != sequence.count(")"):
                return sequence


def generate_unbalanced_sequence(
    length: int,
    rng: random.Random,
    equal_count_probability: float = 0.8,
) -> str:
    """
    Generate an unbalanced example.

    By default:

        80% -> equal-count hard negatives
        20% -> count-mismatch negatives

    This makes shortcuts much less useful while still giving
    the dataset some variety.
    """

    # Equal numbers of '(' and ')' are only possible
    # for an even sequence length.
    if (
        length % 2 == 0
        and rng.random() < equal_count_probability
    ):
        return generate_equal_count_unbalanced(length, rng)

    return generate_count_mismatch_unbalanced(length, rng)


# ---------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------

def choose_length(
    min_length: int,
    max_length: int,
    rng: random.Random,
    require_even: bool = False,
) -> int:
    """
    Randomly select a sequence length.

    Balanced sequences require an even length.
    """

    possible_lengths = list(range(min_length, max_length + 1))

    if require_even:
        possible_lengths = [
            length
            for length in possible_lengths
            if length % 2 == 0
        ]

    if not possible_lengths:
        raise ValueError(
            f"No valid lengths between {min_length} and {max_length}"
        )

    return rng.choice(possible_lengths)


def generate_dataset(
    num_examples: int,
    min_length: int,
    max_length: int,
    seed: int,
    balanced_fraction: float = 0.5,
    hard_negative_fraction: float = 0.8,
) -> List[Tuple[str, int]]:
    """
    Generate a complete binary classification dataset.

    Labels:
        1 = balanced
        0 = unbalanced
    """

    rng = random.Random(seed)

    num_balanced = round(num_examples * balanced_fraction)
    num_unbalanced = num_examples - num_balanced

    examples: List[Tuple[str, int]] = []
    used_sequences: Set[str] = set()

    # -----------------------------------------------------
    # Positive examples
    # -----------------------------------------------------

    while len(examples) < num_balanced:
        length = choose_length(
            min_length,
            max_length,
            rng,
            require_even=True,
        )

        sequence = generate_balanced_sequence(length, rng)

        if sequence in used_sequences:
            continue

        used_sequences.add(sequence)
        examples.append((sequence, 1))

    # -----------------------------------------------------
    # Negative examples
    # -----------------------------------------------------

    negatives_generated = 0

    while negatives_generated < num_unbalanced:
        length = choose_length(
            min_length,
            max_length,
            rng,
            require_even=False,
        )

        sequence = generate_unbalanced_sequence(
            length=length,
            rng=rng,
            equal_count_probability=hard_negative_fraction,
        )

        if sequence in used_sequences:
            continue

        used_sequences.add(sequence)
        examples.append((sequence, 0))

        negatives_generated += 1

    rng.shuffle(examples)

    return examples


# ---------------------------------------------------------
# Dataset saving
# ---------------------------------------------------------

def save_csv(
    dataset: List[Tuple[str, int]],
    path: Path,
) -> None:
    """
    Save dataset as CSV.

    Columns:
        sequence
        label
        length
    """

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "sequence",
                "label",
                "length",
            ]
        )

        for sequence, label in dataset:
            writer.writerow(
                [
                    sequence,
                    label,
                    len(sequence),
                ]
            )


def save_jsonl(
    dataset: List[Tuple[str, int]],
    path: Path,
) -> None:
    """
    Save dataset in JSON Lines format.

    Example:

        {"sequence": "(()())", "label": 1, "length": 6}
    """

    with path.open("w", encoding="utf-8") as file:
        for sequence, label in dataset:
            record = {
                "sequence": sequence,
                "label": label,
                "length": len(sequence),
            }

            file.write(json.dumps(record) + "\n")


# ---------------------------------------------------------
# Dataset diagnostics
# ---------------------------------------------------------

def print_statistics(
    name: str,
    dataset: List[Tuple[str, int]],
) -> None:
    """
    Print useful statistics so we can verify the dataset.
    """

    total = len(dataset)

    balanced = [
        sequence
        for sequence, label in dataset
        if label == 1
    ]

    unbalanced = [
        sequence
        for sequence, label in dataset
        if label == 0
    ]

    equal_count_unbalanced = [
        sequence
        for sequence in unbalanced
        if sequence.count("(") == sequence.count(")")
    ]

    lengths = [len(sequence) for sequence, _ in dataset]

    print(f"\n{name}")
    print("-" * 50)

    print(f"Examples:              {total:,}")

    print(
        f"Balanced:              "
        f"{len(balanced):,} "
        f"({len(balanced) / total:.1%})"
    )

    print(
        f"Unbalanced:            "
        f"{len(unbalanced):,} "
        f"({len(unbalanced) / total:.1%})"
    )

    if unbalanced:
        print(
            f"Hard negatives:        "
            f"{len(equal_count_unbalanced):,} "
            f"({len(equal_count_unbalanced) / len(unbalanced):.1%} "
            f"of negatives)"
        )

    print(f"Minimum length:        {min(lengths)}")
    print(f"Maximum length:        {max(lengths)}")

    print(
        f"Unique sequences:      "
        f"{len(set(sequence for sequence, _ in dataset)):,}"
    )


# ---------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------

def validate_dataset(
    dataset: List[Tuple[str, int]],
) -> None:
    """
    Automatically check labels and data integrity.
    """

    sequences = set()

    for sequence, label in dataset:

        assert set(sequence).issubset({"(", ")"}), (
            f"Invalid character in {sequence}"
        )

        assert sequence not in sequences, (
            f"Duplicate sequence found: {sequence}"
        )

        sequences.add(sequence)

        true_label = int(is_balanced(sequence))

        assert true_label == label, (
            f"Incorrect label for sequence {sequence}: "
            f"expected {true_label}, got {label}"
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main(args):
    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    dataset_configs = {
        "train": {
            "num_examples": args.train_size,
            "min_length": 8,
            "max_length": 32,
            "seed": args.seed,
        },

        "validation": {
            "num_examples": args.validation_size,
            "min_length": 8,
            "max_length": 32,
            "seed": args.seed + 1,
        },

        "test_in_distribution": {
            "num_examples": args.test_size,
            "min_length": 8,
            "max_length": 32,
            "seed": args.seed + 2,
        },

        "test_extrapolation_40_64": {
            "num_examples": args.test_size,
            "min_length": 40,
            "max_length": 64,
            "seed": args.seed + 3,
        },

        "test_extrapolation_80_128": {
            "num_examples": args.test_size,
            "min_length": 80,
            "max_length": 128,
            "seed": args.seed + 4,
        },
    }

    for name, config in dataset_configs.items():

        print(f"\nGenerating {name}...")

        dataset = generate_dataset(
            num_examples=config["num_examples"],
            min_length=config["min_length"],
            max_length=config["max_length"],
            seed=config["seed"],
            balanced_fraction=0.5,
            hard_negative_fraction=args.hard_negative_fraction,
        )

        validate_dataset(dataset)

        save_csv(
            dataset,
            output_directory / f"{name}.csv",
        )

        save_jsonl(
            dataset,
            output_directory / f"{name}.jsonl",
        )

        print_statistics(name, dataset)

    print("\n" + "=" * 50)
    print("Dataset generation complete.")
    print(f"Saved to: {output_directory.resolve()}")
    print("=" * 50)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate balanced-parentheses datasets."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/generated",
        help="Directory where datasets will be saved.",
    )

    parser.add_argument(
        "--train-size",
        type=int,
        default=50_000,
    )

    parser.add_argument(
        "--validation-size",
        type=int,
        default=5_000,
    )

    parser.add_argument(
        "--test-size",
        type=int,
        default=5_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--hard-negative-fraction",
        type=float,
        default=0.8,
        help=(
            "Fraction of unbalanced examples that preserve "
            "equal counts of '(' and ')'."
        ),
    )

    args = parser.parse_args()

    main(args)