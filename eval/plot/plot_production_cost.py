import os
import shutil
import sys
from os.path import abspath, dirname
from pathlib import Path

# python eval/plot/plot_production_cost.py

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
    / "production_cost_by_email_agent_runs.png"
)
THESIS_OUTPUT_PATH = (
    Path(project_root)
    / "Plantilla_TFM_ETSIINF"
    / "include"
    / "production_cost_by_email_agent_runs.png"
)

SERVER_COST_PER_DAY = 0.86
DAYS_PER_MONTH = 30
FIXED_MONTHLY_COST = SERVER_COST_PER_DAY * DAYS_PER_MONTH
EMAIL_AGENT_COST_PER_EMAIL = 0.157
FREE_CREDITS_PER_MONTH = 30.00
MAX_EMAILS_PER_MONTH = 300
BREAK_EVEN_EMAILS = (
    (FREE_CREDITS_PER_MONTH - FIXED_MONTHLY_COST)
    / EMAIL_AGENT_COST_PER_EMAIL
)

def plot_production_cost(output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    emails_per_month = list(range(MAX_EMAILS_PER_MONTH + 1))
    fixed_costs = [FIXED_MONTHLY_COST for _ in emails_per_month]
    variable_costs = [
        EMAIL_AGENT_COST_PER_EMAIL * emails
        for emails in emails_per_month
    ]
    total_costs = [
        fixed_cost + variable_cost
        for fixed_cost, variable_cost in zip(fixed_costs, variable_costs)
    ]

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax.plot(
        emails_per_month,
        total_costs,
        color=lighten_hex_color("#F58518"),
        linewidth=2.6,
        label="Total cost",
    )
    ax.plot(
        emails_per_month,
        variable_costs,
        color=lighten_hex_color("#F53255"),
        linewidth=2.3,
        label="Email-agent cost (\\$0.157/email)",
    )
    ax.plot(
        emails_per_month,
        fixed_costs,
        color=lighten_hex_color("#5B6770"),
        linewidth=2.3,
        label="Fixed server cost (\\$0.86/day ≈ \\$25.80/month)",
    )
    ax.hlines(
        FREE_CREDITS_PER_MONTH,
        xmin=0,
        xmax=MAX_EMAILS_PER_MONTH,
        color=lighten_hex_color("#9DCA1C"),
        linewidth=2.0,
        linestyle="--",
        label="Modal free credits (\\$30/month)",
    )
    ax.scatter(
        [BREAK_EVEN_EMAILS],
        [FREE_CREDITS_PER_MONTH],
        s=42,
        color=lighten_hex_color("#9DCA1C"),
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )

    ax.set_title(
        "Estimated monthly email service cost",
        fontsize=15,
        pad=18,
    )
    ax.set_xlabel("Generated email drafts per month", fontsize=11.5)
    ax.set_ylabel("Monthly cost (USD)", fontsize=11.5)
    ax.set_xlim(0, MAX_EMAILS_PER_MONTH + 5)
    ax.set_ylim(0, max(total_costs) * 1.14)
    ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
    ax.tick_params(axis="both", labelsize=10.5)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.annotate(
        f"Free-tier limit:\n{BREAK_EVEN_EMAILS:.1f} emails/month",
        xy=(BREAK_EVEN_EMAILS, FREE_CREDITS_PER_MONTH),
        xytext=(0, 14),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9.2,
        color="black",
    )

    for emails, label in [(100, "100 emails/month\n≈ 10 emails every 3 days:"), (300, "300 emails/month\n≈ 10 emails daily:")]:
        total_cost = FIXED_MONTHLY_COST + EMAIL_AGENT_COST_PER_EMAIL * emails
        paid_cost = total_cost - FREE_CREDITS_PER_MONTH
        ax.scatter(
            [emails],
            [total_cost],
            s=42,
            color=lighten_hex_color("#F58518"),
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        ax.annotate(
            f"{label}\n\\$({FIXED_MONTHLY_COST:.2f} + {total_cost - FIXED_MONTHLY_COST:.2f} - 30) =\n\\${paid_cost:.2f}/month (to pay)",
            xy=(emails, total_cost),
            xytext=(-12 if emails == 300 else 0, 14),
            textcoords="offset points",
            ha="right" if emails == 300 else "center",
            va="bottom",
            fontsize=9.2,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def main():
    plot_production_cost(OUTPUT_PATH)
    THESIS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUTPUT_PATH, THESIS_OUTPUT_PATH)
    print(f"plot saved: {OUTPUT_PATH.relative_to(project_root)}")
    print(f"thesis copy saved: {THESIS_OUTPUT_PATH.relative_to(project_root)}")

if __name__ == "__main__":
    main()
