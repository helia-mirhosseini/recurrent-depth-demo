import json
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

BASELINE_RESULTS_PATH = Path(
    "results/baseline/evaluation/baseline_test_results.json"
)

RECURRENT_RESULTS_PATH = Path(
    "results/recurrent/evaluation/recurrent_test_results.json"
)

BASELINE_HISTORY_PATH = Path(
    "results/baseline/training_history.json"
)

RECURRENT_HISTORY_PATH = Path(
    "results/recurrent/training_history.json"
)

OUTPUT_DIR = Path(
    "results/figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load JSON
# ============================================================

def load_json(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


baseline_results = load_json(
    BASELINE_RESULTS_PATH
)

recurrent_results = load_json(
    RECURRENT_RESULTS_PATH
)


# ============================================================
# Helper functions
# ============================================================

def save_figure(filename):

    path = OUTPUT_DIR / filename

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Saved: {path}"
    )

    plt.close()


def recurrent_for_range(
    length_range,
):
    """
    Extract all recurrent results belonging to one
    length range and sort by loop count.
    """

    rows = [
        row
        for row in recurrent_results
        if row["length_range"]
        == length_range
    ]

    return sorted(
        rows,
        key=lambda x: x["num_loops"],
    )


# ============================================================
# FIGURE 1
#
# Accuracy vs recurrent loops
# ============================================================

def plot_accuracy_vs_loops():

    plt.figure(
        figsize=(9, 6)
    )

    length_ranges = [
        "8-32",
        "40-64",
        "80-128",
    ]

    for length_range in length_ranges:

        rows = recurrent_for_range(
            length_range
        )

        loops = [
            row["num_loops"]
            for row in rows
        ]

        accuracy = [
            row["accuracy"] * 100
            for row in rows
        ]

        plt.plot(
            loops,
            accuracy,
            marker="o",
            linewidth=2,
            label=f"Recurrent {length_range}",
        )

        # ----------------------------------------------
        # Baseline horizontal reference
        # ----------------------------------------------

        baseline_accuracy = (
            baseline_results[
                length_range
            ]["accuracy"]
            * 100
        )

        plt.axhline(
            baseline_accuracy,
            linestyle="--",
            alpha=0.45,
            label=(
                f"Baseline {length_range} "
                f"({baseline_accuracy:.1f}%)"
            ),
        )

    plt.xlabel(
        "Number of recurrent loops"
    )

    plt.ylabel(
        "Test accuracy (%)"
    )

    plt.title(
        "Does More Recurrent Computation Improve Accuracy?"
    )

    plt.xticks(
        [1, 2, 4, 6, 8]
    )

    plt.ylim(
        45,
        102,
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend(
        fontsize=8
    )

    save_figure(
        "01_accuracy_vs_loops.png"
    )


# ============================================================
# FIGURE 2
#
# Latency vs loops
# ============================================================

def plot_latency_vs_loops():

    plt.figure(
        figsize=(8, 5.5)
    )

    # Use 40-64 as the representative length range.
    #
    # The purpose of this graph is to demonstrate
    # compute scaling as recurrence increases.

    rows = recurrent_for_range(
        "40-64"
    )

    loops = [
        row["num_loops"]
        for row in rows
    ]

    latency = [
        row[
            "latency_ms_per_example"
        ]
        for row in rows
    ]

    plt.plot(
        loops,
        latency,
        marker="o",
        linewidth=2,
    )

    plt.xlabel(
        "Number of recurrent loops"
    )

    plt.ylabel(
        "Latency per example (ms)"
    )

    plt.title(
        "More Internal Computation Has a Cost"
    )

    plt.xticks(
        [1, 2, 4, 6, 8]
    )

    plt.grid(
        alpha=0.25
    )

    save_figure(
        "02_latency_vs_loops.png"
    )


# ============================================================
# FIGURE 3
#
# Baseline vs best recurrent
# ============================================================

def plot_baseline_vs_best_recurrent():

    length_ranges = [
        "8-32",
        "40-64",
        "80-128",
    ]

    baseline_accuracies = []

    recurrent_accuracies = []

    best_loops = []

    for length_range in length_ranges:

        baseline_accuracy = (
            baseline_results[
                length_range
            ]["accuracy"]
            * 100
        )

        baseline_accuracies.append(
            baseline_accuracy
        )

        rows = recurrent_for_range(
            length_range
        )

        best_row = max(
            rows,
            key=lambda x: x["accuracy"],
        )

        recurrent_accuracies.append(
            best_row["accuracy"] * 100
        )

        best_loops.append(
            best_row["num_loops"]
        )

    x = list(
        range(
            len(length_ranges)
        )
    )

    width = 0.35

    plt.figure(
        figsize=(8, 5.5)
    )

    baseline_positions = [
        value - width / 2
        for value in x
    ]

    recurrent_positions = [
        value + width / 2
        for value in x
    ]

    baseline_bars = plt.bar(
        baseline_positions,
        baseline_accuracies,
        width,
        label="4-layer baseline",
    )

    recurrent_bars = plt.bar(
        recurrent_positions,
        recurrent_accuracies,
        width,
        label="Best recurrent configuration",
    )

    plt.xticks(
        x,
        length_ranges,
    )

    plt.xlabel(
        "Sequence length"
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.title(
        "Baseline vs Best Recurrent-Depth Result"
    )

    plt.ylim(
        45,
        102,
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.2,
    )

    # ----------------------------------------------
    # Values above baseline bars
    # ----------------------------------------------

    for bar, accuracy in zip(
        baseline_bars,
        baseline_accuracies,
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,

            bar.get_height() + 0.7,

            f"{accuracy:.1f}%",

            ha="center",
            fontsize=9,
        )

    # ----------------------------------------------
    # Values + loop count above recurrent bars
    # ----------------------------------------------

    for bar, accuracy, loops in zip(
        recurrent_bars,
        recurrent_accuracies,
        best_loops,
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,

            bar.get_height() + 0.7,

            f"{accuracy:.1f}%\n({loops} loop"
            f"{'s' if loops != 1 else ''})",

            ha="center",
            fontsize=9,
        )

    save_figure(
        "03_baseline_vs_recurrent.png"
    )


# ============================================================
# FIGURE 4
#
# Hard-negative performance
# ============================================================

def plot_hard_negative_accuracy():

    plt.figure(
        figsize=(8.5, 5.5)
    )

    length_ranges = [
        "8-32",
        "40-64",
        "80-128",
    ]

    # We focus on recurrent performance here.

    for length_range in length_ranges:

        rows = recurrent_for_range(
            length_range
        )

        loops = [
            row["num_loops"]
            for row in rows
        ]

        accuracies = [
            row[
                "hard_negative_accuracy"
            ]
            * 100
            for row in rows
        ]

        plt.plot(
            loops,
            accuracies,
            marker="o",
            linewidth=2,
            label=length_range,
        )

    plt.xlabel(
        "Number of recurrent loops"
    )

    plt.ylabel(
        "Hard-negative accuracy (%)"
    )

    plt.title(
        "Performance When Counting Alone Cannot Solve the Task"
    )

    plt.xticks(
        [1, 2, 4, 6, 8]
    )

    plt.ylim(
        80,
        101,
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend(
        title="Sequence length"
    )

    save_figure(
        "04_hard_negative_accuracy.png"
    )


# ============================================================
# FIGURE 5
#
# Training curves
# ============================================================

def plot_training_accuracy():

    if (
        not BASELINE_HISTORY_PATH.exists()
        or
        not RECURRENT_HISTORY_PATH.exists()
    ):
        print(
            "Training history files not found. "
            "Skipping training plot."
        )

        return

    baseline_history = load_json(
        BASELINE_HISTORY_PATH
    )

    recurrent_history = load_json(
        RECURRENT_HISTORY_PATH
    )

    baseline_epochs = [
        row["epoch"]
        for row in baseline_history
    ]

    baseline_train = [
        row["train_accuracy"] * 100
        for row in baseline_history
    ]

    baseline_val = [
        row["validation_accuracy"]
        * 100
        for row in baseline_history
    ]

    recurrent_epochs = [
        row["epoch"]
        for row in recurrent_history
    ]

    recurrent_train = [
        row["train_accuracy"] * 100
        for row in recurrent_history
    ]

    recurrent_val = [
        row["validation_accuracy"]
        * 100
        for row in recurrent_history
    ]

    plt.figure(
        figsize=(9, 5.5)
    )

    plt.plot(
        baseline_epochs,
        baseline_train,
        marker="o",
        label="Baseline training",
    )

    plt.plot(
        baseline_epochs,
        baseline_val,
        marker="o",
        label="Baseline validation",
    )

    plt.plot(
        recurrent_epochs,
        recurrent_train,
        marker="o",
        label="Recurrent training",
    )

    plt.plot(
        recurrent_epochs,
        recurrent_val,
        marker="o",
        label="Recurrent validation",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.title(
        "Training and Validation Accuracy"
    )

    plt.xticks(
        baseline_epochs
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    save_figure(
        "05_training_curves.png"
    )


# ============================================================
# FIGURE 6
#
# Balanced vs unbalanced extrapolation
# ============================================================

def plot_balanced_vs_unbalanced():

    rows = recurrent_for_range(
        "40-64"
    )

    loops = [
        row["num_loops"]
        for row in rows
    ]

    balanced = [
        row[
            "balanced_accuracy"
        ]
        * 100
        for row in rows
    ]

    unbalanced = [
        row[
            "unbalanced_accuracy"
        ]
        * 100
        for row in rows
    ]

    plt.figure(
        figsize=(8, 5.5)
    )

    plt.plot(
        loops,
        balanced,
        marker="o",
        linewidth=2,
        label="Balanced sequences",
    )

    plt.plot(
        loops,
        unbalanced,
        marker="o",
        linewidth=2,
        label="Unbalanced sequences",
    )

    plt.xlabel(
        "Number of recurrent loops"
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.title(
        "Where Does the Recurrent Model Fail? (Length 40–64)"
    )

    plt.xticks(
        [1, 2, 4, 6, 8]
    )

    plt.ylim(
        0,
        101,
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    save_figure(
        "06_balanced_vs_unbalanced.png"
    )


# ============================================================
# Run everything
# ============================================================

def main():

    print(
        "Generating experiment figures..."
    )

    plot_accuracy_vs_loops()

    plot_latency_vs_loops()

    plot_baseline_vs_best_recurrent()

    plot_hard_negative_accuracy()

    plot_training_accuracy()

    plot_balanced_vs_unbalanced()

    print()
    print(
        f"Done. Figures saved in: "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()