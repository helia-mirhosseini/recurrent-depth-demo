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

from models.recurrent_transformer import (
    RecurrentTransformer,
    count_parameters,
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
# Train one epoch
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

    # Track how often each loop count occurs.
    loop_usage = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
    }

    start_time = time.perf_counter()

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

        # ----------------------------------------------------
        # Random loop count PER BATCH.
        #
        # This is simpler and much faster than choosing
        # a different loop count for every example.
        # ----------------------------------------------------

        loops = random.randint(
            model.min_train_loops,
            model.max_train_loops,
        )

        loop_usage[loops] += 1

        optimizer.zero_grad()

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_loops=loops,
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_examples += batch_size

    elapsed = (
        time.perf_counter()
        - start_time
    )

    return {
        "loss":
            total_loss
            / total_examples,

        "accuracy":
            total_correct
            / total_examples,

        "time_seconds":
            elapsed,

        "loop_usage":
            loop_usage,
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
    num_loops=4,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    start_time = time.perf_counter()

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

        batch_size = labels.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_examples += batch_size

    elapsed = (
        time.perf_counter()
        - start_time
    )

    return {
        "loss":
            total_loss
            / total_examples,

        "accuracy":
            total_correct
            / total_examples,

        "time_seconds":
            elapsed,
    }


# ============================================================
# Checkpoint
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
            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "validation_accuracy":
                validation_accuracy,

            "config":
                config,
        },
        path,
    )


# ============================================================
# Main
# ============================================================

def main(args):

    set_seed(args.seed)

    device = get_device()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 65)
    print(
        "Recurrent-Depth Transformer Training"
    )
    print("=" * 65)

    print(
        f"Device: {device}"
    )

    print(
        f"Seed:   {args.seed}"
    )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = (
        ParenthesesTokenizer(
            max_length=args.max_length
        )
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = (
        ParenthesesDataset(
            csv_path=args.train_path,
            tokenizer=tokenizer,
        )
    )

    validation_dataset = (
        ParenthesesDataset(
            csv_path=
                args.validation_path,

            tokenizer=tokenizer,
        )
    )

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

    model = RecurrentTransformer(
        vocab_size=
            tokenizer.vocab_size,

        max_length=
            args.max_length,

        d_model=
            args.d_model,

        num_heads=
            args.num_heads,

        dim_feedforward=
            args.dim_feedforward,

        dropout=
            args.dropout,

        num_classes=2,

        min_train_loops=
            args.min_train_loops,

        max_train_loops=
            args.max_train_loops,
    )

    model = model.to(device)

    parameters = count_parameters(
        model
    )

    print(
        f"Trainable parameters: "
        f"{parameters:,}"
    )

    print(
        "Training loop range: "
        f"{args.min_train_loops}"
        f"-"
        f"{args.max_train_loops}"
    )

    # --------------------------------------------------------
    # Loss + optimizer
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=
            args.weight_decay,
    )

    # --------------------------------------------------------
    # Experiment configuration
    # --------------------------------------------------------

    config = {
        "model_type":
            "recurrent",

        "seed":
            args.seed,

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

        "dim_feedforward":
            args.dim_feedforward,

        "dropout":
            args.dropout,

        "max_length":
            args.max_length,

        "min_train_loops":
            args.min_train_loops,

        "max_train_loops":
            args.max_train_loops,

        "validation_loops":
            args.validation_loops,

        "num_parameters":
            parameters,
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
    # Training
    # --------------------------------------------------------

    history = []

    best_accuracy = 0.0

    print(
        "\nStarting training...\n"
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        train_metrics = (
            train_one_epoch(
                model=model,
                dataloader=
                    train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )
        )

        validation_metrics = (
            evaluate(
                model=model,
                dataloader=
                    validation_loader,
                criterion=criterion,
                device=device,
                num_loops=
                    args.validation_loops,
            )
        )

        print(
            f"Epoch "
            f"{epoch:02d}/"
            f"{args.epochs}"
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

        print(
            f"  Loop usage:     "
            f"{train_metrics['loop_usage']}"
        )

        record = {
            "epoch":
                epoch,

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

            "loop_usage":
                train_metrics[
                    "loop_usage"
                ],
        }

        history.append(record)

        # ----------------------------------------------------
        # Save latest
        # ----------------------------------------------------

        save_checkpoint(
            output_dir
            / "recurrent_latest.pt",

            model,
            optimizer,
            epoch,

            validation_metrics[
                "accuracy"
            ],

            config,
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if (
            validation_metrics[
                "accuracy"
            ]
            > best_accuracy
        ):

            best_accuracy = (
                validation_metrics[
                    "accuracy"
                ]
            )

            save_checkpoint(
                output_dir
                / "recurrent_best.pt",

                model,
                optimizer,
                epoch,
                best_accuracy,
                config,
            )

            print(
                f"  ✓ New best checkpoint "
                f"({best_accuracy:.2%})"
            )

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

        print()

    print("=" * 65)

    print(
        "Training complete."
    )

    print(
        f"Best validation accuracy: "
        f"{best_accuracy:.2%}"
    )

    print(
        f"Results saved to: "
        f"{output_dir.resolve()}"
    )

    print("=" * 65)


# ============================================================
# Arguments
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-path",
        type=str,
        default=
            "data/generated/train.csv",
    )

    parser.add_argument(
        "--validation-path",
        type=str,
        default=
            "data/generated/validation.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=
            "results/recurrent",
    )

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

    parser.add_argument(
        "--min-train-loops",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max-train-loops",
        type=int,
        default=4,
    )

    # Use four loops when deciding which checkpoint is best.
    parser.add_argument(
        "--validation-loops",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    main(args)