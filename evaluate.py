import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import (
    ParenthesesDataset,
    ParenthesesTokenizer,
)

from models.baseline_transformer import (
    BaselineTransformer,
    count_parameters,
)


# ============================================================
# Device selection
# ============================================================

def get_device():
    """
    Prefer Apple Silicon MPS, then CUDA, then CPU.
    """

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# Load trained baseline
# ============================================================

def load_model(
    checkpoint_path,
    device,
):
    """
    Load the best baseline checkpoint together with the
    architecture configuration used during training.
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint["config"]

    model = BaselineTransformer(
        vocab_size=4,
        max_length=config["max_length"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        num_classes=2,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    return model, checkpoint


# ============================================================
# Determine negative type
# ============================================================

def is_equal_count_negative(sequence):
    """
    Return True if:

        number of '(' == number of ')'

    Such sequences are particularly important because
    character counting alone cannot classify them.
    """

    return (
        sequence.count("(")
        ==
        sequence.count(")")
    )


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_dataset(
    model,
    dataloader,
    raw_dataset,
    criterion,
    device,
):
    """
    Evaluate a model on one complete test dataset.

    In addition to overall accuracy, calculate:

        balanced accuracy
        unbalanced accuracy
        equal-count hard-negative accuracy
        count-mismatch negative accuracy
    """

    model.eval()

    total_loss = 0.0
    total_examples = 0
    total_correct = 0

    balanced_total = 0
    balanced_correct = 0

    unbalanced_total = 0
    unbalanced_correct = 0

    hard_negative_total = 0
    hard_negative_correct = 0

    easy_negative_total = 0
    easy_negative_correct = 0

    incorrect_examples = []

    dataset_index = 0

    start_time = time.perf_counter()

    for batch in dataloader:

        input_ids = batch[
            "input_ids"
        ].to(device)

        attention_mask = batch[
            "attention_mask"
        ].to(device)

        labels = batch[
            "label"
        ].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        loss = criterion(
            logits,
            labels,
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        batch_size = labels.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_examples += batch_size

        total_correct += (
            predictions == labels
        ).sum().item()

        # Move these tiny tensors back to CPU
        # for per-example analysis.
        predictions_cpu = (
            predictions.cpu().tolist()
        )

        labels_cpu = (
            labels.cpu().tolist()
        )

        # ----------------------------------------------------
        # Analyze every example in the batch
        # ----------------------------------------------------

        for prediction, label in zip(
            predictions_cpu,
            labels_cpu,
        ):

            row = raw_dataset.data.iloc[
                dataset_index
            ]

            sequence = row["sequence"]

            correct = prediction == label

            # ------------------------------------------------
            # Positive / balanced
            # ------------------------------------------------

            if label == 1:

                balanced_total += 1

                if correct:
                    balanced_correct += 1

            # ------------------------------------------------
            # Negative / unbalanced
            # ------------------------------------------------

            else:

                unbalanced_total += 1

                if correct:
                    unbalanced_correct += 1

                # Hard negative:
                # equal number of opening and closing
                # parentheses.
                if is_equal_count_negative(
                    sequence
                ):

                    hard_negative_total += 1

                    if correct:
                        hard_negative_correct += 1

                # Count mismatch negative.
                else:

                    easy_negative_total += 1

                    if correct:
                        easy_negative_correct += 1

            # ------------------------------------------------
            # Save some mistakes for inspection
            # ------------------------------------------------

            if not correct:

                incorrect_examples.append(
                    {
                        "sequence": sequence,
                        "length": len(sequence),
                        "true_label": label,
                        "predicted_label":
                            prediction,
                    }
                )

            dataset_index += 1

    elapsed = (
        time.perf_counter()
        - start_time
    )

    average_latency_ms = (
        elapsed
        / total_examples
        * 1000
    )

    def safe_accuracy(correct, total):

        if total == 0:
            return None

        return correct / total

    return {
        "loss":
            total_loss / total_examples,

        "accuracy":
            total_correct / total_examples,

        "balanced_accuracy":
            safe_accuracy(
                balanced_correct,
                balanced_total,
            ),

        "unbalanced_accuracy":
            safe_accuracy(
                unbalanced_correct,
                unbalanced_total,
            ),

        "hard_negative_accuracy":
            safe_accuracy(
                hard_negative_correct,
                hard_negative_total,
            ),

        "count_mismatch_accuracy":
            safe_accuracy(
                easy_negative_correct,
                easy_negative_total,
            ),

        "num_examples":
            total_examples,

        "hard_negative_examples":
            hard_negative_total,

        "elapsed_seconds":
            elapsed,

        "latency_ms_per_example":
            average_latency_ms,

        "incorrect_examples":
            incorrect_examples,
    }


# ============================================================
# Pretty printing
# ============================================================

def print_results(
    dataset_name,
    metrics,
):

    print()
    print("=" * 65)
    print(dataset_name)
    print("=" * 65)

    print(
        f"Examples:                  "
        f"{metrics['num_examples']:,}"
    )

    print(
        f"Loss:                      "
        f"{metrics['loss']:.4f}"
    )

    print(
        f"Overall accuracy:          "
        f"{metrics['accuracy']:.2%}"
    )

    if (
        metrics["balanced_accuracy"]
        is not None
    ):
        print(
            f"Balanced accuracy:         "
            f"{metrics['balanced_accuracy']:.2%}"
        )

    if (
        metrics["unbalanced_accuracy"]
        is not None
    ):
        print(
            f"Unbalanced accuracy:       "
            f"{metrics['unbalanced_accuracy']:.2%}"
        )

    if (
        metrics["hard_negative_accuracy"]
        is not None
    ):
        print(
            f"Hard-negative accuracy:    "
            f"{metrics['hard_negative_accuracy']:.2%}"
        )

    if (
        metrics["count_mismatch_accuracy"]
        is not None
    ):
        print(
            f"Count-mismatch accuracy:   "
            f"{metrics['count_mismatch_accuracy']:.2%}"
        )

    print(
        f"Hard-negative examples:    "
        f"{metrics['hard_negative_examples']:,}"
    )

    print(
        f"Evaluation time:           "
        f"{metrics['elapsed_seconds']:.3f}s"
    )

    print(
        f"Latency / example:         "
        f"{metrics['latency_ms_per_example']:.4f} ms"
    )

    print(
        f"Misclassified examples:    "
        f"{len(metrics['incorrect_examples']):,}"
    )


# ============================================================
# Save CSV summary
# ============================================================

def save_summary_csv(
    results,
    output_path,
):

    columns = [
        "dataset",
        "num_examples",
        "loss",
        "accuracy",
        "balanced_accuracy",
        "unbalanced_accuracy",
        "hard_negative_accuracy",
        "count_mismatch_accuracy",
        "hard_negative_examples",
        "elapsed_seconds",
        "latency_ms_per_example",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )

        writer.writeheader()

        for dataset_name, metrics in (
            results.items()
        ):

            row = {
                "dataset":
                    dataset_name,

                "num_examples":
                    metrics["num_examples"],

                "loss":
                    metrics["loss"],

                "accuracy":
                    metrics["accuracy"],

                "balanced_accuracy":
                    metrics[
                        "balanced_accuracy"
                    ],

                "unbalanced_accuracy":
                    metrics[
                        "unbalanced_accuracy"
                    ],

                "hard_negative_accuracy":
                    metrics[
                        "hard_negative_accuracy"
                    ],

                "count_mismatch_accuracy":
                    metrics[
                        "count_mismatch_accuracy"
                    ],

                "hard_negative_examples":
                    metrics[
                        "hard_negative_examples"
                    ],

                "elapsed_seconds":
                    metrics[
                        "elapsed_seconds"
                    ],

                "latency_ms_per_example":
                    metrics[
                        "latency_ms_per_example"
                    ],
            }

            writer.writerow(row)


# ============================================================
# Main
# ============================================================

def main(args):

    device = get_device()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 65)
    print("Baseline Evaluation")
    print("=" * 65)

    print(
        f"Device:     {device}"
    )

    print(
        f"Checkpoint: {args.checkpoint}"
    )

    # --------------------------------------------------------
    # Load baseline
    # --------------------------------------------------------

    model, checkpoint = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    config = checkpoint["config"]

    print(
        f"Best validation accuracy: "
        f"{checkpoint['validation_accuracy']:.2%}"
    )

    print(
        f"Checkpoint epoch:         "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Trainable parameters:     "
        f"{count_parameters(model):,}"
    )

    # --------------------------------------------------------
    # Same tokenizer for every dataset
    # --------------------------------------------------------

    tokenizer = ParenthesesTokenizer(
        max_length=config["max_length"]
    )

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Our three important test ranges
    # --------------------------------------------------------

    datasets = {

        "8-32": (
            "data/generated/"
            "test_in_distribution.csv"
        ),

        "40-64": (
            "data/generated/"
            "test_extrapolation_40_64.csv"
        ),

        "80-128": (
            "data/generated/"
            "test_extrapolation_80_128.csv"
        ),
    }

    results = {}

    # --------------------------------------------------------
    # Evaluate all ranges
    # --------------------------------------------------------

    for dataset_name, dataset_path in (
        datasets.items()
    ):

        dataset = ParenthesesDataset(
            csv_path=dataset_path,
            tokenizer=tokenizer,
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        metrics = evaluate_dataset(
            model=model,
            dataloader=loader,
            raw_dataset=dataset,
            criterion=criterion,
            device=device,
        )

        results[
            dataset_name
        ] = metrics

        print_results(
            dataset_name,
            metrics,
        )

        # ----------------------------------------------------
        # Save failure examples
        # ----------------------------------------------------

        failures_path = (
            output_dir
            / f"errors_{dataset_name.replace('-', '_')}.json"
        )

        with open(
            failures_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metrics[
                    "incorrect_examples"
                ],
                file,
                indent=4,
            )

    # --------------------------------------------------------
    # Save main summary
    # --------------------------------------------------------

    save_summary_csv(
        results=results,
        output_path=(
            output_dir
            / "baseline_test_results.csv"
        ),
    )

    # JSON copy without enormous error lists.
    compact_results = {}

    for name, metrics in (
        results.items()
    ):

        compact_results[name] = {
            key: value
            for key, value
            in metrics.items()
            if key != "incorrect_examples"
        }

    with open(
        output_dir
        / "baseline_test_results.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            compact_results,
            file,
            indent=4,
        )

    print()
    print("=" * 65)
    print("Evaluation complete.")
    print(
        f"Results saved to: "
        f"{output_dir.resolve()}"
    )
    print("=" * 65)


# ============================================================
# Arguments
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the ordinary Transformer "
            "baseline on all sequence-length ranges."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "results/baseline/"
            "baseline_best.pt"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "results/baseline/evaluation"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    args = parser.parse_args()

    main(args)