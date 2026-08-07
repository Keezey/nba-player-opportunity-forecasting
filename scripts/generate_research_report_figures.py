"""Generate the figures used by the concise research report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "docs" / "figures"
ROLLING_SUMMARY = ROOT / "reports" / "v2_1_rolling_seasons" / "summary.csv"
POOL_SUMMARY = ROOT / "reports" / "v2_1_pool_size_experiment" / "summary.csv"

STAT_LABELS = {
    "fga": "Field-goal attempts",
    "reb_chances": "Rebound chances",
    "reb": "Rebounds",
}
STAT_COLORS = {
    "fga": "#1F4E79",
    "reb_chances": "#2E7D5B",
    "reb": "#B4552D",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def build_pipeline_figure() -> Path:
    fig, ax = plt.subplots(figsize=(13.5, 4.2))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    boxes = [
        (0.3, 1.45, 1.8, 1.15, "Pregame NBA data", "#E8EEF5"),
        (2.6, 2.45, 2.2, 1.05, "30-day player\nbaseline", "#E8EEF5"),
        (2.6, 0.65, 2.2, 1.05, "Similar-player\nmatchup evidence", "#E8EEF5"),
        (5.35, 1.45, 2.0, 1.15, "V1 opportunity\nprojection", "#F5E9D8"),
        (7.9, 1.45, 2.1, 1.15, "Elastic Net\nresidual correction", "#E4F1E9"),
        (10.55, 2.45, 2.2, 1.05, "Final FGA and\nrebound chances", "#E4F1E9"),
        (10.55, 0.65, 2.2, 1.05, "Learned conversion\nand final rebounds", "#F3E5E1"),
    ]

    for x, y, width, height, label, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.03,rounding_size=0.06",
            linewidth=1.1,
            edgecolor="#39434D",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center")

    arrows = [
        ((2.1, 2.15), (2.6, 2.95)),
        ((2.1, 1.85), (2.6, 1.15)),
        ((4.8, 2.95), (5.35, 2.2)),
        ((4.8, 1.15), (5.35, 1.85)),
        ((7.35, 2.02), (7.9, 2.02)),
        ((10.0, 2.2), (10.55, 2.95)),
        ((10.0, 1.85), (10.55, 1.15)),
        ((11.65, 2.45), (11.65, 1.7)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color="#56616B",
                connectionstyle="arc3,rad=0.0",
            )
        )

    ax.text(
        6.75,
        3.88,
        "All features and tuning inputs are restricted to information available before tip-off",
        ha="center",
        va="center",
        fontsize=10,
        color="#39434D",
    )

    output = FIGURE_DIR / "model_pipeline.png"
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output


def build_rolling_mae_figure() -> Path:
    summary = pd.read_csv(ROLLING_SUMMARY)
    rows = summary[
        (summary["model_type"] == "elastic_net")
        & (summary["scope"] == "all")
        & (summary["aggregation"] == "game_weighted")
    ].copy()

    seasons = ["2023-24", "2024-25", "2025-26"]
    stats = ["fga", "reb_chances", "reb"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    bar_width = 0.36

    for ax, stat in zip(axes, stats):
        stat_rows = rows[rows["stat"] == stat].set_index("test_season").loc[seasons]
        x = list(range(len(seasons)))
        v1 = stat_rows["v1_mae"].to_numpy()
        v2 = stat_rows["v2_mae"].to_numpy()
        ax.bar(
            [value - bar_width / 2 for value in x],
            v1,
            width=bar_width,
            color="#AAB2BA",
            label="V1",
        )
        ax.bar(
            [value + bar_width / 2 for value in x],
            v2,
            width=bar_width,
            color=STAT_COLORS[stat],
            label="V2 Elastic Net",
        )
        for index, (base, corrected) in enumerate(zip(v1, v2)):
            improvement = 100 * (base - corrected) / base
            ax.text(
                index,
                max(base, corrected) + 0.09,
                f"{improvement:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_title(STAT_LABELS[stat])
        ax.set_xticks(x, seasons)
        ax.set_ylim(0, 3.55)
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("Mean absolute error")
    fig.suptitle("Chronological outer-season performance", fontsize=13, y=1.02)
    fig.text(
        0.5,
        0.96,
        "Gray bars: V1 baseline | Colored bars: V2 Elastic Net",
        ha="center",
        fontsize=9,
        color="#39434D",
    )
    fig.text(
        0.5,
        -0.01,
        "Percent labels show V2 improvement over the tuned V1 baseline in each hidden season.",
        ha="center",
        fontsize=9,
    )

    output = FIGURE_DIR / "rolling_outer_season_mae.png"
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output


def build_pool_size_figure() -> Path:
    summary = pd.read_csv(POOL_SUMMARY)
    rows = summary[
        (summary["model_type"] == "elastic_net")
        & (summary["scope"] == "all")
        & (summary["aggregation"] == "game_weighted")
    ].copy()

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for stat in ["fga", "reb_chances", "reb"]:
        stat_rows = rows[rows["stat"] == stat].sort_values("pool_size")
        ax.plot(
            stat_rows["pool_size"],
            stat_rows["mae_improvement_pct"],
            marker="o",
            linewidth=2.0,
            markersize=5,
            color=STAT_COLORS[stat],
            label=STAT_LABELS[stat],
        )

    ax.set_xlabel("Number of target players in the training pool")
    ax.set_ylabel("MAE improvement over V1 (%)")
    ax.set_xticks([30, 60, 90, 150, 250])
    ax.set_ylim(0, 5.2)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    ax.set_title("Effect of training-pool size on Elastic Net residual learning")

    output = FIGURE_DIR / "pool_size_learning_curve.png"
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    outputs = [
        build_pipeline_figure(),
        build_rolling_mae_figure(),
        build_pool_size_figure(),
    ]
    for output in outputs:
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
