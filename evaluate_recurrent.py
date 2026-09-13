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

from models.recurrent_transformer import (
    RecurrentTransformer,
    count_parameters,
)


# ============================================================
# Device
# ============================================================

def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# Load recurrent model
# ============================================================

def load_model(
    checkpoint_path,
    device,
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint["config"]

    model = RecurrentTransformer(
        vocab_size=4,

        max_length=config["max_length"],

        d_model=config["d_model"],

        num_heads=config["num_heads"],

        dim_feedforward=
            config["dim_feedforward"],

        dropout=config["dropout"],

        num_classes=2,

        min_train_loops=
            config["min_train_loops"],

        max_train_loops=
            config["max_train_loops"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    return model, checkpoint


# ============================================================
# Hard-negative check
# ============================================================

def is_equal_count_negative(
    sequence,
):
    """
    Hard negative:

        number '(' == number ')'

    but the sequence is nevertheless invalid.

    Example:

        ())(()
    """

    return (
        sequence.count("(")
        ==
        sequence.count(")")
    )


# ============================================================
# Synchronize device
# ============================================================

def synchronize_device(device):
    """
    GPU operations can be asynchronous.

    Synchronizing before/after timing gives more reliable
    latency measurements.

    This is particularly useful for Apple MPS.
    """

    if device.type == "cuda":
        torch.cuda.synchronize()

    elif device.type == "mps":
        torch.mps.synchronize()


# ============================================================
# Evaluate one dataset at one loop count
# ============================================================

@torch.no_grad()
def evaluate_dataset(
    model,
    dataloader,
    raw_dataset,
    criterion,
    device,
    num_loops,
):

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

    count_mismatch_total = 0
    count_mismatch_correct = 0

    incorrect_examples = []

    dataset_index = 0

    # --------------------------------------------------------
    # Synchronize before timing
    # --------------------------------------------------------

    synchronize_device(device)

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Batches
    # --------------------------------------------------------

    for batch in dataloader:

        input_ids = (
            batch["input_ids"]
            .to(device)
        )

        attention_mask = (
            batch["attention_mask"]
            .to(device)
        )

        labels = (
            batch["label"]
            .to(device)
        )

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_loops=num_loops,
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
            loss.item()
            * batch_size
        )

        total_examples += batch_size

        total_correct += (
            predictions == labels
        ).sum().item()

        # ----------------------------------------------------
        # Move small outputs back to CPU
        # ----------------------------------------------------

        predictions_cpu = (
            predictions
            .cpu()
            .tolist()
        )

        labels_cpu = (
            labels
            .cpu()
            .tolist()
        )

        # ----------------------------------------------------
        # Per-example analysis
        # ----------------------------------------------------

        for prediction, label in zip(
            predictions_cpu,
            labels_cpu,
        ):

            row = (
                raw_dataset
                .data
                .iloc[dataset_index]
            )

            sequence = row["sequence"]

            correct = (
                prediction == label
            )

            # ================================================
            # Balanced
            # ================================================

            if label == 1:

                balanced_total += 1

                if correct:
                    balanced_correct += 1

            # ================================================
            # Unbalanced
            # ================================================

            else:

                unbalanced_total += 1

                if correct:
                    unbalanced_correct += 1

                # --------------------------------------------
                # Equal-count hard negative
                # --------------------------------------------

                if is_equal_count_negative(
                    sequence
                ):

                    hard_negative_total += 1

                    if correct:
                        hard_negative_correct += 1

                # --------------------------------------------
                # Count mismatch
                # --------------------------------------------

                else:

                    count_mismatch_total += 1

                    if correct:
                        count_mismatch_correct += 1

            # ================================================
            # Store mistakes
            # ================================================

            if not correct:

                incorrect_examples.append(
                    {
                        "sequence":
                            sequence,

                        "length":
                            len(sequence),

                        "true_label":
                            label,

                        "predicted_label":
                            prediction,

                        "num_loops":
                            num_loops,
                    }
                )

            dataset_index += 1

    # --------------------------------------------------------
    # Synchronize after inference
    # --------------------------------------------------------

    synchronize_device(device)

    elapsed = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------------
    # Helper
    # --------------------------------------------------------

    def safe_accuracy(
        correct,
        total,
    ):

        if total == 0:
            return None

        return correct / total

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    return {

        "num_loops":
            num_loops,

        "loss":
            total_loss
            / total_examples,

        "accuracy":
            total_correct
            / total_examples,

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
                count_mismatch_correct,
                count_mismatch_total,
            ),

        "num_examples":
            total_examples,

        "hard_negative_examples":
            hard_negative_total,

        "elapsed_seconds":
            elapsed,

        "latency_ms_per_example":
            (
                elapsed
                / total_examples
                * 1000
            ),

        "incorrect_examples":
            incorrect_examples,
    }


# ============================================================
# Warmup
# ============================================================

@torch.no_grad()
def warmup_model(
    model,
    dataloader,
    device,
    num_loops=4,
    batches=3,
):
    """
    Run a few batches before measuring latency.

    This reduces distortion caused by first-call initialization,
    graph compilation, memory allocation, etc.
    """

    model.eval()

    for index, batch in enumerate(
        dataloader
    ):

        if index >= batches:
            break

        input_ids = (
            batch["input_ids"]
            .to(device)
        )

        attention_mask = (
            batch["attention_mask"]
            .to(device)
        )

        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_loops=num_loops,
        )

    synchronize_device(device)


# ============================================================
# Print one result
# ============================================================

def print_result(
    length_range,
    metrics,
):

    print()
    print("-" * 68)

    print(
        f"Length: {length_range} | "
        f"Loops: {metrics['num_loops']}"
    )

    print("-" * 68)

    print(
        f"Overall accuracy:          "
        f"{metrics['accuracy']:.2%}"
    )

    print(
        f"Balanced accuracy:         "
        f"{metrics['balanced_accuracy']:.2%}"
    )

    print(
        f"Unbalanced accuracy:       "
        f"{metrics['unbalanced_accuracy']:.2%}"
    )

    print(
        f"Hard-negative accuracy:    "
        f"{metrics['hard_negative_accuracy']:.2%}"
    )

    print(
        f"Count-mismatch accuracy:   "
        f"{metrics['count_mismatch_accuracy']:.2%}"
    )

    print(
        f"Loss:                      "
        f"{metrics['loss']:.4f}"
    )

    print(
        f"Latency / example:         "
        f"{metrics['latency_ms_per_example']:.4f} ms"
    )

    print(
        f"Errors:                    "
        f"{len(metrics['incorrect_examples']):,}"
    )


# ============================================================
# Save summary CSV
# ============================================================

def save_summary_csv(
    all_results,
    output_path,
):

    columns = [
        "length_range",
        "num_loops",
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

        for result in all_results:

            row = {
                key: result[key]
                for key in columns
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

    print("=" * 68)
    print(
        "Recurrent-Depth Transformer Evaluation"
    )
    print("=" * 68)

    print(
        f"Device:     {device}"
    )

    print(
        f"Checkpoint: {args.checkpoint}"
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    model, checkpoint = load_model(
        checkpoint_path=
            args.checkpoint,

        device=device,
    )

    config = checkpoint["config"]

    print(
        f"Checkpoint epoch:          "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best validation accuracy:  "
        f"{checkpoint['validation_accuracy']:.2%}"
    )

    print(
        f"Trainable parameters:      "
        f"{count_parameters(model):,}"
    )

    print(
        f"Training loops:             "
        f"{config['min_train_loops']}"
        f"-"
        f"{config['max_train_loops']}"
    )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = ParenthesesTokenizer(
        max_length=config["max_length"]
    )

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Test datasets
    # --------------------------------------------------------

    datasets = {

        "8-32":
            "data/generated/"
            "test_in_distribution.csv",

        "40-64":
            "data/generated/"
            "test_extrapolation_40_64.csv",

        "80-128":
            "data/generated/"
            "test_extrapolation_80_128.csv",
    }

    # --------------------------------------------------------
    # Loop counts to test
    # --------------------------------------------------------

    loop_counts = [
        1,
        2,
        4,
        6,
        8,
    ]

    all_results = []

    # ========================================================
    # Evaluate each dataset
    # ========================================================

    for length_range, dataset_path in (
        datasets.items()
    ):

        print()
        print("=" * 68)

        print(
            f"Loading test range: "
            f"{length_range}"
        )

        print("=" * 68)

        dataset = ParenthesesDataset(
            csv_path=dataset_path,
            tokenizer=tokenizer,
        )

        dataloader = DataLoader(
            dataset,

            batch_size=
                args.batch_size,

            shuffle=False,

            num_workers=0,
        )

        # ----------------------------------------------------
        # Warmup once before timing this dataset
        # ----------------------------------------------------

        warmup_model(
            model=model,

            dataloader=dataloader,

            device=device,

            num_loops=4,

            batches=
                args.warmup_batches,
        )

        # ====================================================
        # Test every loop count
        # ====================================================

        for num_loops in loop_counts:

            metrics = evaluate_dataset(
                model=model,

                dataloader=dataloader,

                raw_dataset=dataset,

                criterion=criterion,

                device=device,

                num_loops=num_loops,
            )

            print_result(
                length_range,
                metrics,
            )

            # ------------------------------------------------
            # Flat row for CSV
            # ------------------------------------------------

            result_row = {
                "length_range":
                    length_range,

                **{
                    key: value
                    for key, value
                    in metrics.items()
                    if key
                    != "incorrect_examples"
                },
            }

            all_results.append(
                result_row
            )

            # ------------------------------------------------
            # Save errors separately
            # ------------------------------------------------

            error_file = (
                output_dir
                / (
                    f"errors_"
                    f"{length_range.replace('-', '_')}"
                    f"_loops_{num_loops}.json"
                )
            )

            with open(
                error_file,
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

    # ========================================================
    # Save CSV
    # ========================================================

    save_summary_csv(
        all_results=all_results,

        output_path=(
            output_dir
            / "recurrent_test_results.csv"
        ),
    )

    # ========================================================
    # Save JSON
    # ========================================================

    with open(
        output_dir
        / "recurrent_test_results.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            all_results,
            file,
            indent=4,
        )

    print()
    print("=" * 68)

    print(
        "Recurrent evaluation complete."
    )

    print(
        f"Results saved to: "
        f"{output_dir.resolve()}"
    )

    print("=" * 68)


# ============================================================
# Arguments
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate recurrent-depth model "
            "at multiple loop counts."
        )
    )

    parser.add_argument(
        "--checkpoint",

        type=str,

        default=(
            "results/recurrent/"
            "recurrent_best.pt"
        ),
    )

    parser.add_argument(
        "--output-dir",

        type=str,

        default=(
            "results/recurrent/"
            "evaluation"
        ),
    )

    parser.add_argument(
        "--batch-size",

        type=int,

        default=128,
    )

    parser.add_argument(
        "--warmup-batches",

        type=int,

        default=3,
    )

    args = parser.parse_args()

    main(args)