import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
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
# Reproducibility
# ============================================================

def set_seed(seed: int):
    """
    Set random seeds so experiments are as reproducible
    as reasonably possible.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Device selection
# ============================================================

def get_device():
    """
    Priority:

        Apple Silicon GPU -> MPS
        NVIDIA GPU        -> CUDA
        Otherwise         -> CPU
    """

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# Accuracy
# ============================================================

def calculate_accuracy(
    logits,
    labels,
):
    predictions = torch.argmax(
        logits,
        dim=1,
    )

    correct = (
        predictions == labels
    ).sum().item()

    total = labels.size(0)

    return correct, total


# ============================================================
# Training one epoch
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

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

        # --------------------------------------------
        # Reset gradients
        # --------------------------------------------

        optimizer.zero_grad()

        # --------------------------------------------
        # Forward
        # --------------------------------------------

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        loss = criterion(
            logits,
            labels,
        )

        # --------------------------------------------
        # Backpropagation
        # --------------------------------------------

        loss.backward()

        # Optional but useful for stability.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        # --------------------------------------------
        # Metrics
        # --------------------------------------------

        batch_size = labels.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        correct, total = calculate_accuracy(
            logits,
            labels,
        )

        total_correct += correct
        total_examples += total

    elapsed = time.perf_counter() - start_time

    average_loss = (
        total_loss / total_examples
    )

    accuracy = (
        total_correct / total_examples
    )

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "time_seconds": elapsed,
    }


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

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

        batch_size = labels.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        correct, total = calculate_accuracy(
            logits,
            labels,
        )

        total_correct += correct
        total_examples += total

    elapsed = time.perf_counter() - start_time

    return {
        "loss": (
            total_loss / total_examples
        ),
        "accuracy": (
            total_correct / total_examples
        ),
        "time_seconds": elapsed,
    }


# ============================================================
# Save checkpoint
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    validation_accuracy,
    config,
):

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "validation_accuracy":
                validation_accuracy,

            "config": config,
        },
        path,
    )


# ============================================================
# Main training function
# ============================================================

def main(args):

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(args.seed)

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print("=" * 60)
    print("Baseline Transformer Training")
    print("=" * 60)

    print(f"Device: {device}")
    print(f"Seed:   {args.seed}")

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = ParenthesesTokenizer(
        max_length=args.max_length
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = ParenthesesDataset(
        csv_path=args.train_path,
        tokenizer=tokenizer,
    )

    validation_dataset = ParenthesesDataset(
        csv_path=args.validation_path,
        tokenizer=tokenizer,
    )

    print(
        f"Training examples:   "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation examples: "
        f"{len(validation_dataset):,}"
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = BaselineTransformer(
        vocab_size=tokenizer.vocab_size,
        max_length=args.max_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_classes=2,
    )

    model = model.to(device)

    num_parameters = count_parameters(
        model
    )

    print(
        f"Trainable parameters: "
        f"{num_parameters:,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = {
        "seed": args.seed,
        "device": str(device),

        "batch_size":
            args.batch_size,

        "epochs":
            args.epochs,

        "learning_rate":
            args.learning_rate,

        "weight_decay":
            args.weight_decay,

        "d_model":
            args.d_model,

        "num_heads":
            args.num_heads,

        "num_layers":
            args.num_layers,

        "dim_feedforward":
            args.dim_feedforward,

        "dropout":
            args.dropout,

        "max_length":
            args.max_length,

        "num_parameters":
            num_parameters,
    }

    with open(
        output_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    history = []

    best_validation_accuracy = 0.0

    print("\nStarting training...\n")

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        validation_metrics = evaluate(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch "
            f"{epoch:02d}/{args.epochs}"
        )

        print(
            f"  Train loss:     "
            f"{train_metrics['loss']:.4f}"
        )

        print(
            f"  Train accuracy: "
            f"{train_metrics['accuracy']:.2%}"
        )

        print(
            f"  Val loss:       "
            f"{validation_metrics['loss']:.4f}"
        )

        print(
            f"  Val accuracy:   "
            f"{validation_metrics['accuracy']:.2%}"
        )

        print(
            f"  Train time:     "
            f"{train_metrics['time_seconds']:.2f}s"
        )

        print()

        epoch_record = {
            "epoch": epoch,

            "train_loss":
                train_metrics["loss"],

            "train_accuracy":
                train_metrics[
                    "accuracy"
                ],

            "validation_loss":
                validation_metrics[
                    "loss"
                ],

            "validation_accuracy":
                validation_metrics[
                    "accuracy"
                ],

            "train_time_seconds":
                train_metrics[
                    "time_seconds"
                ],

            "validation_time_seconds":
                validation_metrics[
                    "time_seconds"
                ],
        }

        history.append(
            epoch_record
        )

        # ----------------------------------------------------
        # Save latest model
        # ----------------------------------------------------

        save_checkpoint(
            path=output_dir
            / "baseline_latest.pt",

            model=model,
            optimizer=optimizer,
            epoch=epoch,

            validation_accuracy=
                validation_metrics[
                    "accuracy"
                ],

            config=config,
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if (
            validation_metrics[
                "accuracy"
            ]
            > best_validation_accuracy
        ):

            best_validation_accuracy = (
                validation_metrics[
                    "accuracy"
                ]
            )

            save_checkpoint(
                path=output_dir
                / "baseline_best.pt",

                model=model,
                optimizer=optimizer,
                epoch=epoch,

                validation_accuracy=
                    best_validation_accuracy,

                config=config,
            )

            print(
                f"  ✓ New best checkpoint "
                f"({best_validation_accuracy:.2%})"
            )

        # ----------------------------------------------------
        # Save history after each epoch
        # ----------------------------------------------------

        with open(
            output_dir
            / "training_history.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                history,
                file,
                indent=4,
            )

    print("\n" + "=" * 60)

    print(
        "Training complete."
    )

    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy:.2%}"
    )

    print(
        f"Results saved to: "
        f"{output_dir.resolve()}"
    )

    print("=" * 60)


# ============================================================
# Command-line arguments
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Train the ordinary Transformer "
            "baseline."
        )
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    parser.add_argument(
        "--train-path",
        type=str,
        default=(
            "data/generated/"
            "train.csv"
        ),
    )

    parser.add_argument(
        "--validation-path",
        type=str,
        default=(
            "data/generated/"
            "validation.csv"
        ),
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "results/"
            "baseline"
        ),
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--dim-feedforward",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=129,
    )

    args = parser.parse_args()

    main(args)