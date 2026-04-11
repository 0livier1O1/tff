"""
Temporary script to preview all plots from an existing artifact.
Run from repo root:  python app/_plot_test.py
"""

from pathlib import Path
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.plots import (
    plot_loss_and_regret,
    plot_loss_vs_runtime,
    plot_loss_vs_runtime_per_seed,
    plot_cr_vs_step,
    plot_step_time_breakdown,
    plot_loss_cr_tradeoff,
)

ARTIFACT = Path("artifacts/exp_adam_image_25s_200d")

traces = []
for pol_dir in sorted(ARTIFACT.glob("seed_*/*/traces.csv")):
    seed = int(pol_dir.parent.parent.name.split("_")[1])
    pol_name = pol_dir.parent.name.replace("_", "-")
    df = pd.read_csv(pol_dir)
    df["Policy"] = pol_name
    df["Seed"] = seed
    traces.append(df)

df_rows = pd.concat(traces, ignore_index=True)

plots = {
    "loss_vs_runtime":          plot_loss_vs_runtime(df_rows),
    "loss_vs_runtime_per_seed": plot_loss_vs_runtime_per_seed(df_rows),
    "cr_vs_step":               plot_cr_vs_step(df_rows),
    "step_time_breakdown":      plot_step_time_breakdown(df_rows),
    "loss_cr_tradeoff":         plot_loss_cr_tradeoff(df_rows),
}

for name, fig in plots.items():
    out = Path(f"boss_plot_{name}.png")
    fig.write_image(str(out), width=1200, height=480, scale=2)
    print(f"Saved → {out}")
