import os
import shutil
import sys
from os.path import abspath, dirname
from pathlib import Path

# python eval/plot/plot_wrrf_temperature_weight_distribution.py

project_root = abspath(dirname(dirname(dirname(__file__))))
sys.path.insert(0, project_root)

for env_var, cache_dir_name in [
    ("MPLCONFIGDIR", "mpl-cache"),
    ("XDG_CACHE_HOME", "xdg-cache"),
    ("FONTCONFIG_CACHE", "fontconfig-cache"),
]:
    cache_dir = Path("/private/tmp") / cache_dir_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(env_var, str(cache_dir))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from helpers.data import lighten_hex_color

OUTPUT_PATH = (
    Path(project_root)
    / "eval"
    / "plot"
    / "wrrf_temperature_weight_distribution.png"
)
THESIS_OUTPUT_PATH = (
    Path(project_root)
    / "Plantilla_TFM_ETSIINF"
    / "include"
    / "wrrf_temperature_weight_distribution.png"
)

ENCODERS = ["SPLADE", "BM25", "M3\nsparse", "M3\ndense", "Jina", "Qwen3"]
TEMPERATURE_SERIES = [
    (
        "T = 1.000",
        [0.148, 0.161, 0.181, 0.197, 0.162, 0.150],
        lighten_hex_color("#00CBBF"),
    ),
    (
        "T = 0.091 (selected)",
        [0.026, 0.064, 0.236, 0.577, 0.068, 0.029],
        lighten_hex_color("#9DCA1C"),
    ),
    (
        "T = 0.050",
        [0.003, 0.015, 0.158, 0.804, 0.017, 0.004],
        lighten_hex_color("#FFA600"),
    ),
]

def plot_wrrf_temperature_weight_distribution(output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_values = list(range(len(ENCODERS)))
    bar_width = 0.24
    fig, ax = plt.subplots(figsize=(11.4, 5.4))

    for series_index, (label, weights, color) in enumerate(TEMPERATURE_SERIES):
        offset = (series_index - 1) * bar_width
        bars = ax.bar(
            [x_value + offset for x_value in x_values],
            weights,
            width=bar_width,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=label,
        )
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.012,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.2,
            )

    ax.set_title(
        "WRRF encoder weights under temperature scaling",
        fontsize=15,
        pad=20,
    )
    ax.set_ylabel("Normalized WRRF weight", fontsize=11.5)
    ax.set_xticks(x_values)
    ax.set_xticklabels(ENCODERS, fontsize=10.5)
    ax.set_ylim(0, 0.89)
    ax.tick_params(axis="y", labelsize=10.5)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=10.5, ncol=3, loc="upper left")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def main():
    plot_wrrf_temperature_weight_distribution(OUTPUT_PATH)
    THESIS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUTPUT_PATH, THESIS_OUTPUT_PATH)
    print(f"plot saved: {OUTPUT_PATH.relative_to(project_root)}")
    print(f"thesis copy saved: {THESIS_OUTPUT_PATH.relative_to(project_root)}")

if __name__ == "__main__":
    main()
