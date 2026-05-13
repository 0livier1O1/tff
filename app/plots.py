"""
Reusable Plotly plot functions for the BOSS dashboard.
Each function accepts a df_rows DataFrame (standard traces format) and
returns a plotly.graph_objects.Figure.

Expected columns: step, par_loss, step_loss, current_cr, selected_arm,
  oracle_best_arm, oracle_arm_rank, oracle_best_loss, regret, cum_regret,
  arm_match, pick_time_s, decomp_time_s, gp_fit_time_s, oracle_time_s, step_time_s, Algo, Seed.
"""

import pandas as pd
import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# (colorscale, sample position) per policy — variants of one algorithm share a hue family.
_POLICY_FAMILY = {
    "mabss-greedy": ("Blues",   0.45),
    "mabss-ucb":    ("Blues",   0.60),
    "mabss-exp3":   ("Blues",   0.75),
    "mabss-exp4":   ("Blues",   0.90),
    "boss-ei":      ("Oranges", 0.55),
    "boss-ucb":     ("Oranges", 0.85),
    "tnale":        ("Greys",   0.85),
}
POLICY_COLORS: dict[str, str] = {
    name: pc.sample_colorscale(scale, [pos])[0]
    for name, (scale, pos) in _POLICY_FAMILY.items()
}


def get_policy_color(name: str) -> str:
    if not name:
        return "#888888"
    n = name.lower().replace("_", "-")
    if n in POLICY_COLORS:
        return POLICY_COLORS[n]
    for suffix in ["greedy", "ucb", "exp3", "exp4", "ei"]:
        if n.endswith(suffix):
            for k in POLICY_COLORS:
                if k.endswith(suffix):
                    return POLICY_COLORS[k]
    return "#888888"


def _rgba(color: str, alpha: float) -> str:
    """Convert a '#rrggbb' or 'rgb(r, g, b)' string to 'rgba(r, g, b, alpha)'."""
    if color.startswith("rgb"):
        return color.replace("rgb(", "rgba(")[:-1] + f", {alpha})"
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _mean_std_over_seeds(df_rows, y_col, x_col="step"):
    """Return per-policy (x, mean, lower, upper) aggregated over seeds."""
    result = {}
    for policy in sorted(df_rows["Algo"].unique()):
        pol = df_rows[df_rows["Algo"] == policy].copy().sort_values(["Seed", "step"])
        if x_col == "cum_time_s":
            pol["cum_time_s"] = pol.groupby("Seed")["step_time_s"].cumsum()
        gb = pol.groupby("step")
        mean = gb[y_col].mean()
        x = gb[x_col].mean() if x_col != "step" else mean.index
        result[policy] = (
            x.values,
            mean.values,
            gb[y_col].min().values,
            gb[y_col].max().values,
        )
    return result



def _interpolate_on_time_grid(
    df_rows: pd.DataFrame, policy: str, y_col: str, n_points: int = 300
):
    """
    Zero-order hold interpolation of any step-wise metric onto a shared time
    grid across seeds. Between steps the value is constant (nothing changed),
    so we hold the last observation until the next step fires.
    Grid runs 0 → earliest seed finish to avoid extrapolation.
    """
    pol = df_rows[df_rows["Algo"] == policy].copy().sort_values(["Seed", "step"])
    pol["cum_time_s"] = pol.groupby("Seed")["step_time_s"].cumsum()
    seeds = pol["Seed"].unique()
    t_max = min(pol[pol["Seed"] == s]["cum_time_s"].max() for s in seeds)
    grid = np.linspace(0, t_max, n_points)
    curves = []
    for seed in seeds:
        s = pol[pol["Seed"] == seed].sort_values("cum_time_s")
        idx = np.searchsorted(s["cum_time_s"].values, grid, side="right") - 1
        idx = np.clip(idx, 0, len(s) - 1)
        curves.append(s[y_col].values[idx])
    curves = np.array(curves)
    mean = curves.mean(axis=0)
    return grid, mean, curves.min(axis=0), curves.max(axis=0)


# ---------------------------------------------------------------------------
# Moved from dashboard.py — these are the canonical dashboard plot functions
# ---------------------------------------------------------------------------


def plot_objective(
    df_boss: pd.DataFrame | None,
    df_tnale: pd.DataFrame | None,
    max_evals: int | None = None,
    n_points: int = 300,
) -> go.Figure:
    """
    Cumulative-min objective (CR + λ·RSE) for BOSS and TnALE on shared axes.

    The x-axis is the per-seed function-evaluation count (one row = one
    decomposition). Legacy rows missing step_time_s are dropped. Optional
    max_evals crops the eval axis.

      Left  — vs function evaluations (min–max band over seeds)
      Right — vs cumulative wall-clock runtime (zero-order hold)
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Cumulative Min Objective vs. Function Evaluations (min–max band)",
            "Cumulative Min Objective vs. Cumulative Runtime (min–max band)",
        ),
    )

    df = pd.concat([df_boss, df_tnale], ignore_index=True)
    df = df[df["step_time_s"].notna() & df["objective"].notna()].copy()
    # Count every recorded row as a step so init samples line up across algos
    # (TNALE's sparsity-guard rows have huge objective but never beat cum-min,
    # so they don't distort the curve; they only add ~0.08s each to cum_time).
    df["n_evals"] = df.groupby(["Algo", "Seed"], sort=False).cumcount() + 1
    if max_evals is not None:
        df = df[df["n_evals"] <= max_evals]

    for policy in sorted(df["Algo"].unique()):
        color = get_policy_color(policy)
        rgb = _rgba(color, 0.2)
        pol = df[df["Algo"] == policy].copy().sort_values(["Seed", "n_evals"])
        pol["cum_min_obj"] = pol.groupby("Seed")["objective"].cummin()
        pol["cum_time_s"] = pol.groupby("Seed")["step_time_s"].cumsum()

        # Left: vs n_evals — aggregate over seeds
        gb = pol.groupby("n_evals")["cum_min_obj"]
        evals = list(gb.groups.keys())
        mean_obj = gb.mean().values
        lo_obj, hi_obj = gb.min().values, gb.max().values

        fig.add_trace(go.Scatter(
            x=evals, y=hi_obj, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=evals, y=lo_obj, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=rgb, showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=evals, y=mean_obj, mode="lines",
            line=dict(color=color, width=2),
            name=policy.upper(), showlegend=False,
        ), row=1, col=1)

        # Right: vs cumulative runtime — zero-order hold interpolation
        seeds = pol["Seed"].unique()
        t_max = min(pol[pol["Seed"] == s]["cum_time_s"].max() for s in seeds)
        grid = np.linspace(0, t_max, n_points)
        curves = []
        for seed in seeds:
            s = pol[pol["Seed"] == seed].sort_values("cum_time_s")
            idx = np.clip(np.searchsorted(s["cum_time_s"].values, grid, side="right") - 1, 0, len(s) - 1)
            curves.append(s["cum_min_obj"].values[idx])
        curves = np.array(curves)
        m_rt, lo_rt, hi_rt = curves.mean(axis=0), curves.min(axis=0), curves.max(axis=0)

        fig.add_trace(go.Scatter(
            x=grid, y=hi_rt, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=grid, y=lo_rt, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=rgb, showlegend=False, hoverinfo="skip",
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=grid, y=m_rt, mode="lines",
            line=dict(color=color, width=2),
            name=policy.upper(), showlegend=True,
        ), row=1, col=2)

    # Mark end of BOSS init phase on both subplots.
    boss_init = df[df["Algo"].str.startswith("boss-") & (df["phase"] == "init")]
    if not boss_init.empty:
        _init_line = dict(color="#888888", width=1, dash="dot")
        _init_anno = dict(size=9, color="#888888")
        fig.add_vline(
            x=int(boss_init["n_evals"].max()),
            line=_init_line, row=1, col=1,
            annotation_text="end of init",
            annotation_position="top",
            annotation_font=_init_anno,
        )
        # Runtime axis: mean total init wall time across BOSS (Algo, Seed) pairs.
        t_init = boss_init.groupby(["Algo", "Seed"])["step_time_s"].sum().mean()
        fig.add_vline(
            x=float(t_init),
            line=_init_line, row=1, col=2,
            annotation_text="end of init",
            annotation_position="top",
            annotation_font=_init_anno,
        )

    fig.update_xaxes(title_text="Function Evaluations", row=1, col=1)
    fig.update_xaxes(title_text="Cumulative Runtime (s)", row=1, col=2)
    fig.update_yaxes(title_text="Cumulative Min Objective", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Min Objective", row=1, col=2)
    fig.update_layout(
        height=380,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def plot_loss_and_regret(df_rows: pd.DataFrame, n_points: int = 300) -> go.Figure:
    """
    2×2 subplot sharing one legend:
      Row 1 — loss and regret vs search step (± 1 std dev over seeds)
      Row 2 — loss and regret vs cumulative wall-clock time (zero-order hold,
               ± 1 std dev over seeds)
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Loss vs. Step (min–max band)",
            "Regret vs. Step (min–max band)",
            "Loss vs. Cumulative Runtime (min–max band)",
            "Regret vs. Cumulative Runtime (min–max band)",
        ),
        vertical_spacing=0.18,
    )

    for policy in df_rows["Algo"].unique():
        color = get_policy_color(policy)
        rgb = _rgba(color, 0.2)
        sub = df_rows[df_rows["Algo"] == policy]
        gb = sub.groupby("step")
        steps = list(gb.groups.keys())

        mean_loss = gb["step_loss"].mean()
        std_loss = gb["step_loss"].std()
        # min_loss = gb["step_loss"].min()
        # max_loss = gb["step_loss"].max()
        mean_regret = gb["cum_regret"].mean()
        min_regret = gb["cum_regret"].min()
        max_regret = gb["cum_regret"].max()

        t_loss, m_loss, lo_loss, hi_loss = _interpolate_on_time_grid(
            df_rows, policy, "step_loss", n_points
        )
        t_regret, m_regret, lo_regret, hi_regret = _interpolate_on_time_grid(
            df_rows, policy, "cum_regret", n_points
        )

        def _band(x, lo, hi, row, col):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=hi,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=lo,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=rgb,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

        # Row 1 — vs step
        # _band(steps, min_loss, max_loss, 1, 1)
        b = 2
        _band(steps, mean_loss - b * std_loss, mean_loss + b * std_loss, 1, 1)
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=mean_loss,
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=False,
                name=policy.upper(),
            ),
            row=1,
            col=1,
        )

        # _band(steps, min_regret, max_regret, 1, 2)
        _band(steps, mean_regret - b * std_loss, mean_regret + b * std_loss, 1, 2)
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=mean_regret,
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=False,
                name=policy.upper(),
            ),
            row=1,
            col=2,
        )

        # Row 2 — vs runtime (legend shown here once per policy)
        _band(t_loss, lo_loss, hi_loss, 2, 1)
        fig.add_trace(
            go.Scatter(
                x=t_loss,
                y=m_loss,
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=False,
                name=policy.upper(),
            ),
            row=2,
            col=1,
        )

        _band(t_regret, lo_regret, hi_regret, 2, 2)
        fig.add_trace(
            go.Scatter(
                x=t_regret,
                y=m_regret,
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=True,
                name=policy.upper(),
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        height=620,
        template="plotly_white",
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Search Step", row=1, col=1)
    fig.update_xaxes(title_text="Search Step", row=1, col=2)
    fig.update_xaxes(title_text="Cumulative Runtime (s)", row=2, col=1)
    fig.update_xaxes(title_text="Cumulative Runtime (s)", row=2, col=2)
    fig.update_yaxes(title_text="Normalized Loss", row=1, col=1)
    fig.update_yaxes(title_text="Regret", row=1, col=2)
    fig.update_yaxes(title_text="Normalized Loss", row=2, col=1)
    fig.update_yaxes(title_text="Regret", row=2, col=2)
    return fig


def plot_arm_trace(sub: pd.DataFrame, algo_name: str, color: str) -> go.Figure:
    """
    Single-policy arm selection trace for one seed: chosen arm vs oracle best
    arm vs oracle arm rank−1. Matches the per-seed trace vector in the dashboard.
    """
    steps_v = sub["step"].values
    chosen_arm = sub["selected_arm"].values
    greedy_oracle = sub["oracle_best_arm"].values
    rank_minus = (
        (sub["oracle_arm_rank"].values - 1)
        if "oracle_arm_rank" in sub.columns
        else np.zeros(len(sub))
    )
    hit_rate = sub["arm_match"].mean()
    unique_arms = len(set(chosen_arm))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps_v,
            y=greedy_oracle,
            mode="lines",
            line=dict(color="#222222", width=1.5, shape="hv"),
            name="Greedy oracle arm",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=steps_v,
            y=chosen_arm,
            mode="lines+markers",
            marker=dict(symbol="x", size=7, color=color),
            line=dict(color=color, width=2),
            name="Chosen arm",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=steps_v,
            y=rank_minus,
            mode="lines",
            line=dict(color="#AAAAAA", width=2, dash="dash"),
            name="Oracle arm rank − 1",
        )
    )
    fig.update_layout(
        title=dict(
            text=f"{algo_name.upper()} — hit rate={hit_rate:.2f}, unique arms={unique_arms}",
            font=dict(size=12),
        ),
        height=300,
        margin=dict(l=0, r=0, t=100, b=0),
        template="plotly_white",
        legend=dict(orientation="h", y=1.25, font=dict(size=10)),
        yaxis=dict(title="Arm id / rank−1", dtick=1),
        xaxis=dict(title="Search step", tickvals=steps_v.tolist()),
    )
    return fig


# ---------------------------------------------------------------------------
# New plot functions (computational cost & tradeoff analysis)
# ---------------------------------------------------------------------------


def plot_loss_vs_runtime_seed(seed_df: pd.DataFrame) -> go.Figure:
    """
    Loss vs. cumulative wall-clock runtime for a single seed, all policies
    overlaid on one figure. seed_df must already be filtered to one seed.
    """
    fig = go.Figure()
    for policy in sorted(seed_df["Algo"].unique()):
        c = get_policy_color(policy)
        pol = seed_df[seed_df["Algo"] == policy].sort_values("step").copy()
        pol["cum_time_s"] = pol["step_time_s"].cumsum()
        fig.add_trace(
            go.Scatter(
                x=pol["cum_time_s"].values,
                y=pol["step_loss"].values,
                mode="lines",
                line=dict(color=c, width=2, shape="hv"),
                name=policy.upper(),
            )
        )
    fig.update_layout(
        title=dict(text="Loss vs. Cumulative Runtime", font=dict(size=12)),
        xaxis_title="Cumulative Wall-Clock Time (s)",
        yaxis_title="Normalized Loss",
        height=300,
        margin=dict(l=0, r=0, t=60, b=0),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", x=1, xanchor="right", y=1.25, font=dict(size=10)),
    )
    return fig


def plot_loss_vs_runtime(df_rows: pd.DataFrame, n_points: int = 300) -> go.Figure:
    """
    Mean post-step loss vs. cumulative wall-clock runtime, aggregated over seeds
    using zero-order hold interpolation onto a shared time grid.
    """
    fig = go.Figure()
    for policy in sorted(df_rows["Algo"].unique()):
        x, mean, lo, hi = _interpolate_on_time_grid(
            df_rows, policy, "step_loss", n_points
        )
        c = get_policy_color(policy)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=hi,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=lo,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=_rgba(c, 0.15),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mean,
                mode="lines",
                line=dict(color=c, width=2),
                name=policy.upper(),
            )
        )
    fig.update_layout(
        title="Mean Post-Step Loss vs. Cumulative Runtime (aggregated over seeds)",
        xaxis_title="Cumulative Wall-Clock Time (s)",
        yaxis_title="Normalized Loss",
        height=380,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.22),
        margin=dict(l=40, r=20, t=50, b=80),
    )
    return fig


def plot_loss_vs_runtime_per_seed(df_rows: pd.DataFrame) -> go.Figure:
    """
    Loss vs. cumulative wall-clock runtime, one line per seed per policy.
    Policy color is shared; seeds are distinguished by decreasing opacity.
    """
    fig = go.Figure()
    for policy in sorted(df_rows["Algo"].unique()):
        c = get_policy_color(policy)
        pol = df_rows[df_rows["Algo"] == policy].copy().sort_values(["Seed", "step"])
        pol["cum_time_s"] = pol.groupby("Seed")["step_time_s"].cumsum()
        seeds = sorted(pol["Seed"].unique())
        n_seeds = len(seeds)
        for i, seed in enumerate(seeds):
            s = pol[pol["Seed"] == seed].sort_values("cum_time_s")
            alpha = 1.0 - 0.5 * (i / max(n_seeds - 1, 1))
            fig.add_trace(
                go.Scatter(
                    x=s["cum_time_s"].values,
                    y=s["step_loss"].values,
                    mode="lines",
                    line=dict(color=_rgba(c, alpha), width=1.5),
                    name=f"{policy.upper()} seed {seed}",
                    legendgroup=policy,
                    showlegend=True,
                )
            )
    fig.update_layout(
        title="Post-Step Loss vs. Cumulative Runtime (per seed)",
        xaxis_title="Cumulative Wall-Clock Time (s)",
        yaxis_title="Normalized Loss",
        height=380,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.22),
        margin=dict(l=40, r=20, t=50, b=80),
    )
    return fig


def plot_cr_vs_step(df_rows: pd.DataFrame) -> go.Figure:
    """Mean compression ratio over search steps."""
    fig = go.Figure()
    for policy, (x, mean, lo, hi) in _mean_std_over_seeds(
        df_rows, "current_cr"
    ).items():
        c = get_policy_color(policy)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=hi,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=lo,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=_rgba(c, 0.15),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mean,
                mode="lines",
                line=dict(color=c, width=2),
                name=policy.upper(),
            )
        )
    fig.update_layout(
        title="Compression Ratio vs. Search Step",
        xaxis_title="Search Step",
        yaxis_title="Compression Ratio (CR)",
        height=380,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.22),
        margin=dict(l=40, r=20, t=50, b=80),
    )
    return fig


def plot_step_time_breakdown(df_rows: pd.DataFrame) -> go.Figure:
    """Stacked bar of per-step time components (decomp, pick, oracle), one panel per policy."""
    policies = sorted(df_rows["Algo"].unique())
    fig = make_subplots(
        rows=1,
        cols=len(policies),
        subplot_titles=[p.upper() for p in policies],
        shared_yaxes=True,
    )
    components = [
        ("oracle_time_s", "Oracle eval", "#AECDE8"),
        ("gp_fit_time_s", "GP fit",      "#F28E2B"),
        ("pick_time_s",   "Arm pick",    "#5A9FC9"),
        ("decomp_time_s", "Decomp fit",  "#1F6FA5"),
    ]
    legend_shown = {label: False for _, label, _ in components}
    for col_i, policy in enumerate(policies, 1):
        is_greedy = policy.lower().replace("mabss-", "") == "greedy"
        pol_df = df_rows[df_rows["Algo"] == policy].sort_values(["Seed", "step"])
        gb = pol_df.groupby("step")
        steps = gb["step"].first().values
        for col_key, label, color in components:
            # Oracle time is only a real cost for greedy; skip for others
            if col_key == "oracle_time_s" and not is_greedy:
                continue
            # Fall back gracefully for legacy traces missing new columns
            if col_key not in pol_df.columns:
                continue
            show = not legend_shown[label]
            if show:
                legend_shown[label] = True
            fig.add_trace(
                go.Bar(
                    x=steps,
                    y=gb[col_key].mean().values,
                    name=label,
                    marker_color=color,
                    showlegend=show,
                ),
                row=1,
                col=col_i,
            )
    fig.update_layout(
        barmode="stack",
        title="Step Time Breakdown (mean over seeds)",
        height=380,
        template="plotly_white",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.22),
        margin=dict(l=40, r=20, t=50, b=80),
    )
    fig.update_yaxes(title_text="Time (s)", col=1)
    fig.update_xaxes(title_text="Search Step")
    return fig


def plot_loss_cr_tradeoff(df_rows: pd.DataFrame) -> go.Figure:
    """Final loss vs. final CR scatter per seed, coloured by policy."""
    fig = go.Figure()
    for policy in sorted(df_rows["Algo"].unique()):
        c = get_policy_color(policy)
        last = (
            df_rows[df_rows["Algo"] == policy]
            .sort_values("step")
            .groupby("Seed")
            .last()
            .reset_index()
        )
        fig.add_trace(
            go.Scatter(
                x=last["current_cr"],
                y=last["step_loss"],
                mode="markers",
                marker=dict(
                    color=c,
                    size=12,
                    symbol="circle",
                    line=dict(color="white", width=1.5),
                ),
                name=policy.upper(),
                text=[f"Seed {s}" for s in last["Seed"]],
                hovertemplate="%{text}<br>CR=%{x:.1f}<br>Loss=%{y:.4f}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Final Loss vs. Compression Ratio (per seed)",
        xaxis_title="Compression Ratio (CR)",
        yaxis_title="Final Normalized Loss",
        height=380,
        template="plotly_white",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.22),
        margin=dict(l=40, r=20, t=50, b=80),
    )
    return fig


def plot_decomp_curves(decomp_data: list[dict], algo_name: str, color: str) -> go.Figure:
    """Concatenated decomposition loss across all search steps.

    Single continuous curve over global epoch index; vertical dashed lines mark
    each rank increment with a label showing the step number and selected arm.
    """
    all_x, all_y = [], []
    step_boundaries = []  # (global_epoch_at_start, step, arm)
    epoch_cursor = 0

    for entry in decomp_data:
        losses = entry["losses"]
        step_boundaries.append((epoch_cursor + 1, entry["step"], entry["arm"]))
        all_x.extend(range(epoch_cursor + 1, epoch_cursor + len(losses) + 1))
        all_y.extend(losses)
        epoch_cursor += len(losses)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=all_x,
            y=all_y,
            mode="lines",
            line=dict(color=color, width=1.5),
            showlegend=False,
            hovertemplate="Epoch %{x}<br>RSE %{y:.5f}<extra></extra>",
        )
    )

    shapes, annotations = [], []
    for x0, step, arm in step_boundaries:
        shapes.append(dict(
            type="line",
            x0=x0, x1=x0, y0=0, y1=1,
            yref="paper",
            line=dict(color=_rgba(color, 0.4), width=1, dash="dash"),
        ))
        annotations.append(dict(
            x=x0, y=1, yref="paper",
            text=f"s{step}<br>a{arm}",
            showarrow=False,
            font=dict(size=9, color=_rgba(color, 0.8)),
            xanchor="left",
            yanchor="top",
        ))

    fig.update_layout(
        title=dict(text=f"{algo_name.upper()} — decomposition convergence", font=dict(size=12)),
        xaxis_title="Global epoch",
        yaxis_title="RSE",
        yaxis_type="log",
        height=320,
        template="plotly_white",
        shapes=shapes,
        annotations=annotations,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def plot_time_to_threshold(
    df_rows: pd.DataFrame, threshold: float = 0.05, y_metric: str = "CR"
) -> go.Figure:
    """Scatter: cumulative runtime vs CR or Efficiency at first step where step_loss <= threshold.

    Each point is one (policy, seed) pair. Colored by policy.
    Points where the threshold is never reached are omitted.
    """
    use_efficiency = y_metric == "Efficiency" and "efficiency" in df_rows.columns
    y_col = "efficiency" if use_efficiency else "current_cr"
    y_label = "Efficiency (CR / target CR)" if use_efficiency else "Compression Ratio (CR)"

    # Cumulative runtime per (Policy, Seed) in step order
    df = df_rows.sort_values(["Algo", "Seed", "step"]).copy()
    df["cum_time"] = df.groupby(["Algo", "Seed"])["step_time_s"].cumsum()

    keep_cols = ["Algo", "Seed", "step", "step_loss", "current_cr", "cum_time"]
    if use_efficiency:
        keep_cols.append("efficiency")

    # First row per (Policy, Seed) where loss hits threshold
    hits = (
        df[df["step_loss"] <= threshold]
        .groupby(["Algo", "Seed"], sort=False)
        .first()
        .reset_index()[keep_cols]
    )

    if hits.empty:
        return go.Figure().update_layout(
            title=f"No policy reached loss ≤ {threshold}",
            template="plotly_white", height=420,
        )

    policies = sorted(hits["Algo"].unique())

    hits["label"] = hits.apply(
        lambda r: (
            f"<b>{r['Policy'].upper()}</b> — Seed {r['Seed']}<br>"
            f"Runtime: {r['cum_time']:.1f}s<br>"
            f"CR: {r['current_cr']:.3f}"
            + (f"<br>Efficiency: {r['efficiency']:.3f}" if use_efficiency else "")
            + f"<br>Step: {int(r['step'])}<br>Loss: {r['step_loss']:.5f}"
        ), axis=1
    )

    fig = go.Figure()

    for policy in policies:
        p = hits[hits["Algo"] == policy]
        sizes = 9 + (p["step_loss"] / threshold) * 10
        fig.add_trace(go.Scatter(
            x=p["cum_time"], y=p[y_col],
            mode="markers",
            marker=dict(
                color=get_policy_color(policy),
                size=sizes.tolist(),
                line=dict(color="white", width=1),
            ),
            name=policy.upper(),
            text=p["label"].tolist(),
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.add_hline(
        y=1,
        line_dash="dash",
        line_color="gray",
        opacity=0.6,
        annotation_text="y = 1",
        annotation_position="right",
        annotation_font=dict(color="gray", size=11),
    )

    fig.update_layout(
        title=dict(
            text=f"Runtime vs {y_metric}",
            font=dict(size=13),
        ),
        xaxis_title="Cumulative runtime at threshold (s)",
        yaxis_title=y_label,
        height=420,
        template="plotly_white",
        legend=dict(orientation="v", x=1.02, y=1, xanchor="left"),
        margin=dict(l=40, r=160, t=50, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# GP-UCB geometry visualisations (matplotlib, notebook-friendly)
# ---------------------------------------------------------------------------

def plot_tn_arms_gp_overlay_plotly(
    A,
    max_rank: int,
    mu,
    sigma=None,
    *,
    normalize: str = "rank",
    title: str | None = None,
    cmap: str = "Viridis",
    node_color: str = "#e6e6e6",
    width: int = 672,
    height: int = 672,
    plot_acqf: bool = False,
    beta: float = 1.0,
) -> go.Figure:
    """
    Plotly version of `plot_tn_arms_gp_overlay` for dashboard comparison.

    NetworkX is not needed for rendering: the same circular TN layout is
    converted directly into Plotly line, marker, and annotation traces.
    """
    import math

    # Normalize inputs to NumPy arrays so masking and ranking use predictable shapes.
    A_np = np.asarray(A).astype(int)
    N = A_np.shape[0]
    mu = np.asarray(mu, dtype=float)

    # Split upper-triangular bonds into admissible arms and saturated bonds.
    arm_edges, saturated_edges = [], []
    full_mask = []
    for i in range(N):
        for j in range(i + 1, N):
            if A_np[i, j] < max_rank:
                arm_edges.append((i, j))
                full_mask.append(True)
            else:
                full_mask.append(False)
                if A_np[i, j] == max_rank:
                    saturated_edges.append((i, j))
    full_mask = np.asarray(full_mask)

    # Keep posterior values only for currently admissible arms.
    if mu.size == full_mask.size:
        mu = mu[full_mask]
    elif mu.size != len(arm_edges):
        raise ValueError(f"mu has length {mu.size}, expected {full_mask.size} or {len(arm_edges)}.")
    if sigma is None:
        if plot_acqf:
            raise ValueError("sigma is required when plot_acqf=True.")
    else:
        sigma = np.asarray(sigma, dtype=float)
        if sigma.size == full_mask.size:
            sigma = sigma[full_mask]
        elif sigma.size != len(arm_edges):
            raise ValueError(
                f"sigma has length {sigma.size}, expected {full_mask.size} or {len(arm_edges)}."
            )

    # Scale Plotly point and line sizes from the requested pixel dimensions.
    linear_scale = float(np.sqrt((width * height) / (672.0 * 672.0)))
    linear_scale = float(np.clip(linear_scale, 0.48, 1.35))

    node_size = 50 * linear_scale
    core_label_size = max(12, 18 * linear_scale)
    edge_label_size = max(12, 17 * linear_scale)
    title_size = max(14, 18 * linear_scale)
    legend_font_size = max(12, 13 * linear_scale)
    cbar_font_size = max(12, 13 * linear_scale)

    # Place core nodes on a fixed circle.
    pos = {}
    for i in range(N):
        angle = math.pi / 2 + 2 * math.pi * i / N
        pos[f"C{i}"] = np.array([math.cos(angle), math.sin(angle)])

    # Map GP means to color ranks so close posterior values remain visible.
    K = len(arm_edges)
    if K > 1:
        mu_sorted = np.sort(mu)
        color_values = np.argsort(np.argsort(mu)) / (K - 1)
        colorbar_values = color_values
        colorbar_cmin, colorbar_cmax = 0.0, 1.0
        max_ticks = max(3, min(K, int(round(height / 120))))
        if K <= max_ticks:
            tick_idx = np.arange(K)
        else:
            tick_idx = np.unique(np.linspace(0, K - 1, max_ticks).round().astype(int))
        colorbar_tickvals = tick_idx / max(K - 1, 1)
    elif K == 1:
        mu_sorted = np.asarray(mu)
        color_values = np.array([0.5])
        colorbar_values = color_values
        colorbar_cmin, colorbar_cmax = 0.0, 1.0
        tick_idx = np.array([0])
        colorbar_tickvals = np.array([0.5])
    else:
        mu_sorted = np.asarray([])
        color_values = np.asarray([])
        colorbar_values = color_values
        colorbar_cmin, colorbar_cmax = 0.0, 1.0
        tick_idx = np.asarray([], dtype=int)
        colorbar_tickvals = np.asarray([])

    # Convert GP uncertainty into edge widths; smaller sigma means thicker edge.
    if sigma is not None and normalize == "rank" and K > 1:
        s_pos = np.argsort(np.argsort(sigma)) / (K - 1)
    elif sigma is not None and K:
        s_pos = (sigma - sigma.min()) / max(sigma.max() - sigma.min(), 1e-12)
    else:
        s_pos = np.zeros(K)
    widths = (8.0 - 6.5 * s_pos) * linear_scale if sigma is not None else np.full(K, 4.0 * linear_scale)

    def _sample_color(value: float) -> str:
        return pc.sample_colorscale(cmap, [float(value)])[0]

    plotly_cmap = cmap
    if plot_acqf:
        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.78, 0.22],
            vertical_spacing=0,
        )
    else:
        fig = go.Figure()

    def _add_graph_trace(trace):
        if plot_acqf:
            fig.add_trace(trace, row=1, col=1)
        else:
            fig.add_trace(trace)

    def _add_graph_annotation(**kwargs):
        if plot_acqf:
            fig.add_annotation(row=1, col=1, **kwargs)
        else:
            fig.add_annotation(**kwargs)

    # Draw saturated bonds separately because they have no GP overlay.
    for i, j in saturated_edges:
        x0, y0 = pos[f"C{i}"]
        x1, y1 = pos[f"C{j}"]
        _add_graph_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(color="black", width=max(1, 2.0 * linear_scale), dash="dot"),
            # hovertemplate=f"C{i}-C{j}<br>rank={max_rank}<br>saturated<extra></extra>",
            showlegend=False,
        ))

    # Draw admissible arm edges with GP mean color and sigma width.
    for idx, (i, j) in enumerate(arm_edges):
        x0, y0 = pos[f"C{i}"]
        x1, y1 = pos[f"C{j}"]
        sigma_text = "" if sigma is None else f"<br>sigma={sigma[idx]:.4g}"
        _add_graph_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(color=_sample_color(color_values[idx]), width=float(widths[idx])),
            # hovertemplate=(
            #     f"C{i}-C{j}<br>rank={A_np[i, j]}"
            #     f"<br>mu={mu[idx]:.4g}{sigma_text}<extra></extra>"
            # ),
            showlegend=False,
        ))

    # Draw core nodes above all edge traces.
    node_names = [f"C{i}" for i in range(N)]
    _add_graph_trace(go.Scatter(
        x=[pos[n][0] for n in node_names],
        y=[pos[n][1] for n in node_names],
        mode="markers+text",
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(color="#4d4d4d", width=max(1, 2 * linear_scale)),
        ),
        text=[f"𝒢<sub>{i + 1}</sub>" for i in range(N)],
        textposition="middle center",
        textfont=dict(size=core_label_size, color="black"),
        # hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))

    # Add an invisible marker trace solely to host the GP mean colorbar.
    if K:
        graph_domain = fig.layout.yaxis.domain if plot_acqf else (0.0, 1.0)
        mid_x = [(pos[f"C{i}"][0] + pos[f"C{j}"][0]) / 2 for i, j in arm_edges]
        mid_y = [(pos[f"C{i}"][1] + pos[f"C{j}"][1]) / 2 for i, j in arm_edges]
        colorbar_x = 0.96
        colorbar_y = float((graph_domain[0] + graph_domain[1]) / 2)
        colorbar_len = 0.58 * float(graph_domain[1] - graph_domain[0])
        colorbar_title_gap = 0.015
        colorbar = dict(
            tickfont=dict(size=cbar_font_size),
            thickness=max(8, int(14 * linear_scale)),
            len=colorbar_len,
            x=colorbar_x,
            y=colorbar_y,
            yanchor="middle",
        )
        if colorbar_tickvals is not None:
            colorbar["tickvals"] = colorbar_tickvals.tolist()
            colorbar["ticktext"] = [f"{float(v):.7g}" for v in mu_sorted[tick_idx]]
        _add_graph_trace(go.Scatter(
            x=mid_x,
            y=mid_y,
            mode="markers",
            marker=dict(
                color=colorbar_values,
                colorscale=cmap,
                cmin=colorbar_cmin,
                cmax=colorbar_cmax,
                size=0.1,
                opacity=0.001,
                colorbar=colorbar,
            ),
            hoverinfo="skip",
            showlegend=False,
        ))
        fig.add_annotation(
            x=colorbar_x,
            y=colorbar_y + colorbar_len / 2 + colorbar_title_gap,
            xref="paper",
            yref="paper",
            text=r"$\mu(\phi(s,a))$",
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=cbar_font_size, color="black"),
        )

    # Label admissible ranks and saturated Rmax bonds at edge midpoints.
    for i, j in arm_edges:
        x = (pos[f"C{i}"][0] + pos[f"C{j}"][0]) / 2
        y = (pos[f"C{i}"][1] + pos[f"C{j}"][1]) / 2
        _add_graph_annotation(
            x=x, y=y,
            text=str(A_np[i, j]),
            showarrow=False,
            font=dict(size=edge_label_size, color="black"),
            bgcolor="rgba(255,255,255,0.55)",
            borderpad=1,
        )
    for i, j in saturated_edges:
        x = (pos[f"C{i}"][0] + pos[f"C{j}"][0]) / 2
        y = (pos[f"C{i}"][1] + pos[f"C{j}"][1]) / 2
        _add_graph_annotation(
            x=x, y=y,
            text="R<sub>max</sub>",
            showarrow=False,
            font=dict(size=edge_label_size, color="black"),
            bgcolor="rgba(255,255,255,0.55)",
            borderpad=1,
        )

    # Add dummy legend traces to explain the sigma-to-width encoding.
    if sigma is not None and K:
        s_lo, s_hi = float(sigma.min()), float(sigma.max())
        _add_graph_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color="dimgray", width=max(1, 1.5 * linear_scale)),
            name=f"high sigma ({s_hi:.2g})",
        ))
        _add_graph_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color="dimgray", width=max(2, 5.5 * linear_scale)),
            name=f"low sigma ({s_lo:.2g})",
        ))

    if plot_acqf:
        ucb = mu + beta * sigma
        best_idx = int(np.argmax(ucb)) if len(ucb) else -1
        bar_colors = ["#0f766e" if k == best_idx else "#b7c7c9" for k in range(len(ucb))]
        edge_labels = [f"({i + 1},{j + 1})" for i, j in arm_edges]
        fig.add_trace(
            go.Bar(
                x=edge_labels,
                y=ucb,
                marker=dict(color=bar_colors, line=dict(color="#4b5563", width=0.4)),
                # hovertemplate="edge %{x}<br>UCB=%{y:.4g}<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        if len(ucb):
            lo, hi = float(ucb.min()), float(ucb.max())
            pad = 0.03 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 1e-3
            fig.update_yaxes(range=[lo - pad, hi + pad], row=2, col=1)
        acqf_font_size = max(10, 12 * linear_scale)
        fig.update_xaxes(
            title_text="edges",
            title_font=dict(size=acqf_font_size),
            tickfont=dict(size=acqf_font_size),
            domain=[0.16, 0.78],
            row=2,
            col=1,
        )
        fig.update_yaxes(
            title_text="",
            tickfont=dict(size=acqf_font_size),
            row=2,
            col=1,
        )
        acqf_domain = fig.layout.yaxis2.domain
        fig.add_annotation(
            x=0.16,
            y=float(acqf_domain[1]) + 0.01,
            xref="paper",
            yref="paper",
            text="UCB values",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=acqf_font_size, color="black"),
        )

    # Hide axes and keep equal aspect so the circular graph is not distorted.
    fig.update_layout(
        # title=dict(text=title or "GP-UCB arms over current TN", font=dict(size=title_size)),
        template="plotly_white",
        width=width,
        height=height,
        margin=dict(l=2, r=8, t=8, b=2),
        legend=dict(
            x=0.98,
            y=1,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
            font=dict(size=legend_font_size),
        ),
        hovermode="closest",
    )
    if plot_acqf:
        fig.update_layout(bargap=0.35)
    if plot_acqf:
        fig.update_xaxes(visible=False, range=[-1.14, 1.22], constrain="domain", row=1, col=1)
        fig.update_yaxes(visible=False, range=[-1.18, 1.18], scaleanchor="x", scaleratio=1, row=1, col=1)
    else:
        fig.update_xaxes(visible=False, range=[-1.14, 1.22], constrain="domain")
        fig.update_yaxes(visible=False, range=[-1.18, 1.18], scaleanchor="x", scaleratio=1)
    return fig


def plot_tensor_network_example(
    A=None,
    edge: tuple[int, int] | None = None,
    cur_rank=None,
    *,
    max_rank: int,
    ax=None,
    figsize: tuple[float, float] = (7, 7),
    edge_index_base: int = 1,
    edge_color: str = "#777777",
    node_color: str = "#e6e6e6",
    node_edge_color: str = "#4d4d4d",
    edge_width: float = 2.2,
    node_size: float = 0.18,
):
    """
    Draw a neutral tensor-network schematic from an adjacency matrix.

    By default this draws the five-core complete example from the paper figure.
    Off-diagonal entries in [1, max_rank] create virtual bonds; larger ranks
    and diagonal physical legs are intentionally omitted. Non-highlighted edge
    labels show A[i, j]. If `edge=(i, j)` and `cur_rank` are supplied, the
    highlighted bond is labelled as a rank increment.
    """
    import math
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    if A is None:
        A_np = np.ones((5, 5), dtype=int)
    else:
        A_np = np.asarray(A).astype(int)
    N = A_np.shape[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    selected_edge = None
    if edge is not None:
        selected_edge = tuple(sorted((edge[0] - edge_index_base, edge[1] - edge_index_base)))
        if (
            len(selected_edge) != 2
            or selected_edge[0] == selected_edge[1]
            or selected_edge[0] < 0
            or selected_edge[1] >= N
        ):
            raise ValueError(
                f"edge={edge!r} is invalid for a tensor network with {N} cores "
                f"and edge_index_base={edge_index_base}."
            )

    selected_nodes = set() if selected_edge is None else set(selected_edge)

    with plt.rc_context({"mathtext.fontset": "cm", "font.family": "serif"}):
        # Layout mirrors the TN overlay: cores sit on a circle with G1 at top.
        pos = {}
        for i in range(N):
            angle = math.pi / 2 + 2 * math.pi * i / N
            pos[i] = np.array([math.cos(angle), math.sin(angle)])

        def _draw_solid_edge(p0, p1, *, lw=edge_width, alpha=1.0, zorder=1):
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                color=edge_color,
                lw=lw,
                alpha=alpha,
                solid_capstyle="round",
                zorder=zorder,
            )

        def _draw_edge_rank_label(p0, p1, label, *, alpha=1.0, zorder=3):
            midpoint = 0.5 * (p0 + p1)
            ax.text(
                midpoint[0],
                midpoint[1],
                str(label),
                ha="center",
                va="center",
                fontsize=18,
                color="black",
                alpha=alpha,
                zorder=zorder,
            )

        # Draw virtual bonds from the upper triangle of A.
        selected_midpoint = None
        for i in range(N):
            for j in range(i + 1, N):
                bond = (i, j)
                rank = int(A_np[i, j])
                if rank <= 0 or rank > max_rank:
                    continue
                p0, p1 = pos[i], pos[j]
                is_selected = selected_edge is not None and bond == selected_edge
                alpha = 1.0 if selected_edge is None or is_selected else 0.16
                zorder = 2 if is_selected else 1
                _draw_solid_edge(p0, p1, alpha=alpha, zorder=zorder)
                if is_selected:
                    selected_midpoint = 0.5 * (p0 + p1)
                else:
                    _draw_edge_rank_label(p0, p1, rank, alpha=alpha, zorder=zorder + 1)

        # Draw neutral tensor cores and core labels.
        for i, p in pos.items():
            alpha = 1.0 if selected_edge is None or i in selected_nodes else 0.16
            zorder = 4 if selected_edge is not None and i in selected_nodes else 3
            ax.add_patch(Circle(
                p,
                node_size,
                facecolor=node_color,
                edgecolor=node_edge_color,
                lw=1.0,
                alpha=alpha,
                zorder=zorder,
            ))
            ax.text(
                p[0],
                p[1],
                rf"$\mathcal{{G}}_{{{i + 1}}}$",
                ha="center",
                va="center",
                fontsize=28,
                color="black",
                alpha=alpha,
                zorder=zorder + 1,
            )

        # Label only the highlighted rank update.
        if selected_midpoint is not None and cur_rank is not None:
            ax.text(
                selected_midpoint[0],
                selected_midpoint[1],
                rf"$R_{{{edge[0]}, {edge[1]}}}={cur_rank}\rightarrow {cur_rank + 1}$",
                ha="center",
                va="center",
                fontsize=22,
                color="black",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.3),
                zorder=5,
            )

        pad = node_size + 0.25
        ax.set_aspect("equal")
        ax.set_xlim(-1 - pad, 1 + pad)
        ax.set_ylim(-1 - pad, 1 + pad)
        ax.axis("off")
        fig.tight_layout(pad=0)
    return fig, ax


def plot_gp_oracle_reward_calibration(
    pol_diagnostics_dict,
    algo: str = "mabss-ucb",
    seed: int = 1,
    step: int | str = "all",
    *,
    std_scale: float = 2.0,
    title: str | None = None,
    width: int = 560,
    height: int = 430,
    slider: bool = False,
) -> go.Figure:
    """
    Compare GP posterior mean against oracle arm rewards for one (seed, algo).
    Each point is one admissible arm at one step, colored by arm index. GP σ is
    shown as a horizontal error bar around the predicted mean.

    step: "all" (default) for every step, or an int to subset to that step.
    slider: when True, render every step as its own group of traces with a
        Plotly-native step slider. Ignored when `step` is a specific integer.
    """
    all_records = pol_diagnostics_dict[(seed, algo)]

    # Global range over every step so the axes don't jump as the slider moves.
    full_mu = np.concatenate([np.asarray(r["gp_mean"], dtype=float) for r in all_records])
    full_oracle = np.concatenate([np.asarray(r["oracle_rewards"], dtype=float) for r in all_records])
    full_valid = np.isfinite(full_mu)
    lo = float(min(np.nanmin(full_mu[full_valid]), np.nanmin(full_oracle[full_valid])))
    hi = float(max(np.nanmax(full_mu[full_valid]), np.nanmax(full_oracle[full_valid])))
    pad = 0.04 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 1e-3

    K = len(all_records[0]["oracle_rewards"])
    N = int((1 + np.sqrt(1 + 8 * K)) / 2)
    triu_i, triu_j = np.triu_indices(N, k=1)
    arm_label = {k: f"({int(triu_i[k]) + 1},{int(triu_j[k]) + 1})" for k in range(K)}
    arm_color = dict(zip(
        range(K),
        pc.sample_colorscale("Portland", [k / max(K - 1, 1) for k in range(K)]),
    ))

    def _extract(records):
        """Stack (mu, oracle, std, steps, arms, chosen) across given records."""
        mu_all, oracle_all, std_all, step_all, arm_all, chosen_all = [], [], [], [], [], []
        for rec in records:
            mu = np.asarray(rec["gp_mean"], dtype=float)
            oracle = np.asarray(rec["oracle_rewards"], dtype=float)
            valid = np.isfinite(mu)
            std_valid = np.asarray(rec["gp_std"], dtype=float)
            std = np.full_like(mu, np.nan)
            if std_valid.size == valid.sum():
                std[valid] = std_valid
            idx = np.flatnonzero(valid)
            mu_all.append(mu[idx])
            oracle_all.append(oracle[idx])
            std_all.append(std[idx])
            step_all.append(np.full(idx.size, rec.get("step", -1), dtype=int))
            arm_all.append(idx.astype(int))
            chosen_all.append(idx == int(rec.get("arm", -1)))
        return (
            np.concatenate(mu_all) if mu_all else np.array([]),
            np.concatenate(oracle_all) if oracle_all else np.array([]),
            np.concatenate(std_all) if std_all else np.array([]),
            np.concatenate(step_all) if step_all else np.array([], dtype=int),
            np.concatenate(arm_all) if arm_all else np.array([], dtype=int),
            np.concatenate(chosen_all).astype(bool) if chosen_all else np.array([], dtype=bool),
        )

    def _add_group(fig, records, *, visible=True, legend_for_arm=None):
        """Add the K per-arm + 1 chosen-stars traces for one record subset.
        Returns the list of indices of traces just added. `legend_for_arm` may
        be a set of arm indices for which to show legend entries (used so the
        slider doesn't duplicate the per-arm legend across step groups)."""
        mu, oracle, s, steps_, arms, chosen = _extract(records)
        err_avail = np.isfinite(s).any() if s.size else False
        added: list[int] = []
        legend_for_arm = legend_for_arm if legend_for_arm is not None else set(range(K))
        for k in range(K):
            sel = (arms == k) & ~chosen
            if not sel.any():
                # Add a stub trace so the slider can toggle the same number of
                # traces per step.
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=10, symbol="circle", color=arm_color[k], opacity=0.85,
                                line=dict(color="white", width=0.4)),
                    name=arm_label[k], legendgroup=f"arm-{k}",
                    showlegend=(k in legend_for_arm),
                    visible=visible, hoverinfo="skip",
                ))
                added.append(len(fig.data) - 1)
                continue
            sx = s[sel]
            fig.add_trace(go.Scatter(
                x=mu[sel], y=oracle[sel], mode="markers",
                marker=dict(size=10, symbol="circle", color=arm_color[k], opacity=0.85,
                            line=dict(color="white", width=0.4)),
                error_x=dict(type="data", array=sx * std_scale, thickness=0.6,
                             width=0, color=arm_color[k])
                        if err_avail and np.isfinite(sx).any() else None,
                name=arm_label[k], legendgroup=f"arm-{k}",
                showlegend=(k in legend_for_arm),
                visible=visible,
                text=[f"arm {arm_label[k]}<br>step={st}<br>μ={my:.4g}<br>oracle={ox:.4g}"
                      + (f"<br>σ={sv:.4g}" if np.isfinite(sv) else "")
                      for st, my, ox, sv in zip(steps_[sel], mu[sel], oracle[sel], sx)],
                hovertemplate="%{text}<extra></extra>",
            ))
            added.append(len(fig.data) - 1)

        # Chosen-stars trace (always present, even if empty, for stable indexing).
        if chosen.any():
            sx = s[chosen]
            fig.add_trace(go.Scatter(
                x=mu[chosen], y=oracle[chosen], mode="markers",
                marker=dict(
                    size=10, symbol="star",
                    color=[arm_color[a] for a in arms[chosen]],
                    line=dict(color="black", width=0.6),
                ),
                error_x=dict(type="data", array=sx * std_scale, thickness=0.6,
                             width=0, color="rgba(0,0,0,0.45)")
                        if err_avail and np.isfinite(sx).any() else None,
                text=[f"arm {arm_label[int(a)]} (selected)<br>step={st}<br>μ={my:.4g}<br>oracle={ox:.4g}"
                      + (f"<br>σ={sv:.4g}" if np.isfinite(sv) else "")
                      for a, st, my, ox, sv in zip(arms[chosen], steps_[chosen], mu[chosen], oracle[chosen], sx)],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False, visible=visible,
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, symbol="star", color="rgba(0,0,0,0.55)",
                            line=dict(color="black", width=0.6)),
                showlegend=False, visible=visible, hoverinfo="skip",
            ))
        added.append(len(fig.data) - 1)
        return added

    fig = go.Figure()

    use_slider = slider and step == "all"
    if use_slider:
        # One group of (K + 1) traces per step; only the first group is initially visible.
        step_values = sorted({r.get("step", -1) for r in all_records})
        group_traces: list[list[int]] = []
        for i, sv in enumerate(step_values):
            recs = [r for r in all_records if r.get("step") == sv]
            visible = (i == 0)
            # Only first group contributes per-arm legend entries.
            legend_for_arm = set(range(K)) if i == 0 else set()
            group_traces.append(_add_group(fig, recs, visible=visible, legend_for_arm=legend_for_arm))
    else:
        records = (
            [r for r in all_records if r.get("step") == int(step)]
            if step != "all" else all_records
        )
        _add_group(fig, records)

    # Color-agnostic "selected arm" legend stub — always visible.
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=10, symbol="star", color="rgba(0,0,0,0.55)",
                    line=dict(color="black", width=0.6)),
        name="selected arm", showlegend=True,
    ))
    legend_idx = len(fig.data) - 1

    # Diagonal y=x — always visible.
    fig.add_trace(go.Scatter(
        x=[lo - pad, hi + pad], y=[lo - pad, hi + pad],
        mode="lines", line=dict(color="rgba(0,0,0,0.55)", width=1, dash="dash"),
        hoverinfo="skip", showlegend=False,
    ))
    diag_idx = len(fig.data) - 1

    layout_kwargs = dict(
        template="plotly_white",
        width=width, height=height,
        margin=dict(l=55, r=20, t=35 if title else 15, b=70 if use_slider else 50),
        xaxis=dict(title="GP mean μ ± σ", range=[lo - pad, hi + pad], showgrid=False, zeroline=False),
        yaxis=dict(title="Oracle reward", range=[lo - pad, hi + pad], showgrid=False, zeroline=False),
        legend=dict(
            font=dict(size=11), itemsizing="constant",
            bgcolor="rgba(255,255,255,0)", borderwidth=0,
            tracegroupgap=0, itemwidth=30,
            yanchor="top", y=1.0, xanchor="left", x=1.005,
        ),
    )
    if title is not None:
        layout_kwargs["title"] = dict(text=title)
    fig.update_layout(**layout_kwargs)

    if use_slider:
        n_traces = len(fig.data)
        slider_steps = []
        for i, sv in enumerate(step_values):
            visible = [False] * n_traces
            for ti in group_traces[i]:
                visible[ti] = True
            visible[legend_idx] = True
            visible[diag_idx] = True
            label_step = sv if sv >= 0 else i
            slider_steps.append(dict(
                method="update",
                label=str(label_step),
                args=[
                    {"visible": visible},
                    {"title.text": (
                        f"{title} — step {label_step}" if title else f"step {label_step}"
                    )},
                ],
            ))
        fig.update_layout(sliders=[dict(
            active=0, pad=dict(t=30, b=10),
            currentvalue=dict(prefix="step = ", font=dict(size=12)),
            steps=slider_steps,
        )])

    return fig


_EXP4_EXPERT_NAMES = ["uniform", "GP-mean", "GP-UCB", "empirical", "recency"]


def _arm_labels_from_K(K: int) -> list[str]:
    N = int((1 + np.sqrt(1 + 8 * K)) / 2)
    triu_i, triu_j = np.triu_indices(N, k=1)
    return [f"({int(triu_i[k]) + 1},{int(triu_j[k]) + 1})" for k in range(K)]


def _arm_heatmap(steps, values, selected_arms, *, value_name: str, visible_mask=None):
    """One arm-distribution heatmap with zero-probability arms left blank."""
    K = values.shape[1]
    y_labels = _arm_labels_from_K(K)

    # Mark saturated arms as NaN so they render as empty cells on the white background.
    if visible_mask is None:
        visible_mask = values > 0
    z = np.where(visible_mask, values, np.nan).T

    main = go.Heatmap(
        z=z, x=steps, y=y_labels,
        colorscale="Viridis",
        zmin=0.0, zmax=float(max(np.nanmax(z), 1e-12)),
        colorbar=dict(thickness=10, len=0.85),
        hovertemplate=f"step %{{x}}<br>arm %{{y}}<br>{value_name}=%{{z:.3g}}<extra></extra>",
        hoverongaps=False,
        ygap=1,
    )

    # Overlay the arm chosen by the policy at each step.
    overlay = go.Scatter(
        x=steps, y=[y_labels[i] for i in selected_arms],
        mode="markers",
        marker=dict(size=6, color="black", symbol="circle-open",
                    line=dict(color="black", width=1.2)),
        name="selected arm", hoverinfo="skip", showlegend=True,
    )
    return main, overlay, y_labels


def plot_policy_weights(
    pol_diagnostics_dict,
    algo: str,
    seed: int = 1,
    *,
    arm_value: str = "probability",
    title: str | None = None,
    width: int = 720,
    height: int = 320,
) -> go.Figure:
    """
    Heatmap of an EXP3/EXP4 policy's distribution evolution.

    EXP3 → arm-distribution heatmap.
    EXP4 → two heatmaps: expert weights (top) + arm distribution (bottom).
    The selected arm at each step is overlaid as a black open circle.
    """
    records = pol_diagnostics_dict[(seed, algo)]
    steps = [r["step"] for r in records]
    selected_arms = [int(r["arm"]) for r in records]
    if arm_value not in {"probability", "weight"}:
        raise ValueError("arm_value must be 'probability' or 'weight'")
    value_name = "p" if arm_value == "probability" else "w"

    if algo.endswith("exp3"):
        probs = np.array([r["exp3_probs"] for r in records], dtype=float)
        if arm_value == "weight":
            log_w = np.array([r["exp3_log_weights"] for r in records], dtype=float)
            centered = log_w - log_w.max(axis=1, keepdims=True)
            values = np.exp(centered)
            values = values / values.sum(axis=1, keepdims=True)
        else:
            values = probs
        main, overlay, _ = _arm_heatmap(
            steps, values, selected_arms,
            value_name=value_name,
            visible_mask=probs > 0,
        )
        fig = go.Figure([main, overlay])
        fig.update_layout(yaxis=dict(title="arm (edge)", autorange="reversed"))
    elif algo.endswith("exp4"):
        ew = np.array([r["expert_weights"] for r in records], dtype=float)
        probs = np.array([r["exp4_probs"] for r in records], dtype=float)
        if arm_value == "weight":
            expert_dists = np.array([r["exp4_expert_dists"] for r in records], dtype=float)
            values = np.einsum("se,sek->sk", ew, expert_dists)
        else:
            values = probs
        main, overlay, _ = _arm_heatmap(
            steps, values, selected_arms,
            value_name=value_name,
            visible_mask=probs > 0,
        )

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.30, 0.70], vertical_spacing=0.06,
        )
        fig.add_trace(go.Heatmap(
            z=ew.T, x=steps, y=_EXP4_EXPERT_NAMES[: ew.shape[1]],
            colorscale="Cividis", zmin=0.0, zmax=1.0,
            colorbar=dict(thickness=10, len=0.30, y=0.85, yanchor="middle"),
            hovertemplate="step %{x}<br>expert %{y}<br>w=%{z:.3g}<extra></extra>",
            ygap=1,
        ), row=1, col=1)
        main.update(colorbar=dict(thickness=10, len=0.55, y=0.30, yanchor="middle"))
        fig.add_trace(main, row=2, col=1)
        fig.add_trace(overlay, row=2, col=1)
        fig.update_yaxes(title_text="expert", autorange="reversed", row=1, col=1)
        fig.update_yaxes(title_text="arm (edge)", autorange="reversed", row=2, col=1)
        height = max(height, 460)
    else:
        raise ValueError(f"plot_policy_weights only supports EXP3/EXP4, got {algo!r}")

    fig.update_layout(
        title=None, template="plotly_white",
        width=width, height=height,
        margin=dict(l=70, r=95, t=8, b=45),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=13),
        legend=dict(
            x=1,
            y=1.0,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            borderwidth=0,
            font=dict(size=11),
            itemsizing="constant",
        ),
    )
    fig.update_yaxes(
        showgrid=False,
        tickfont=dict(size=12),
        title_font=dict(size=13),
        title_standoff=4,
    )
    fig.update_xaxes(
        title_text="Function evals",
        showgrid=False,
        tickfont=dict(size=12),
        title_font=dict(size=13),
        title_standoff=4,
    )
    return fig


def plot_arm_feature_space(
    phi_hist,
    r_hist,
    phi_candidates,
    mu_candidates=None,
    *,
    method: str = "pca",
    ax=None,
    title: str | None = None,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (7, 6),
):
    """
    Option B — 2D projection of (state, arm) feature vectors.

    Past evaluations are plotted as filled circles coloured by their realised
    reward; admissible arms at the current step are plotted as crosses
    coloured by GP posterior mean (or grey if `mu_candidates` is None). The
    projection (PCA by default, UMAP if installed and requested) is fit on
    the union of past and candidate features so both sit in the same frame.

    Parameters
    ----------
    phi_hist        : (N, d) historical feature vectors
    r_hist          : (N,)   realised rewards
    phi_candidates  : (K, d) admissible (state, arm) feature vectors at step t
    mu_candidates   : (K,)   GP posterior mean per candidate (optional)
    method          : "pca" or "umap"
    """
    import matplotlib.pyplot as plt

    phi_hist = np.asarray(phi_hist, dtype=float)
    phi_cand = np.asarray(phi_candidates, dtype=float)
    r_hist = np.asarray(r_hist, dtype=float)

    X = np.vstack([phi_hist, phi_cand])
    if method == "umap":
        import umap  # type: ignore
        Z = umap.UMAP(n_components=2, random_state=0).fit_transform(X)
    else:
        from sklearn.decomposition import PCA
        Z = PCA(n_components=2, random_state=0).fit_transform(X)
    Z_hist, Z_cand = Z[:len(phi_hist)], Z[len(phi_hist):]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sc_hist = ax.scatter(
        Z_hist[:, 0], Z_hist[:, 1],
        c=r_hist, cmap=cmap, s=70, edgecolors="white", linewidths=0.8,
        label="evaluated arms", zorder=2,
    )
    if mu_candidates is not None:
        mu_candidates = np.asarray(mu_candidates, dtype=float)
        ax.scatter(
            Z_cand[:, 0], Z_cand[:, 1],
            c=mu_candidates, cmap=cmap,
            marker="x", s=180, linewidths=2.5,
            label="admissible arms (GP mean)", zorder=3,
        )
    else:
        ax.scatter(
            Z_cand[:, 0], Z_cand[:, 1],
            color="dimgray", marker="x", s=180, linewidths=2.5,
            label="admissible arms", zorder=3,
        )
    cbar = fig.colorbar(sc_hist, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("reward (history) / $\\mu$ (candidates)", fontsize=10)

    ax.set_title(title or f"Feature space ({method.upper()}) — past rewards + admissible arms",
                 fontsize=13)
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.legend(loc="best", fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_pareto_at_step(
    df_rows: pd.DataFrame, step: int, y_metric: str = "CR"
) -> go.Figure:
    """Pareto-style scatter: loss vs CR/Efficiency at a fixed search step.

    One point per (policy, seed). Points per policy are connected sorted by loss.
    """
    use_efficiency = y_metric == "Efficiency" and "efficiency" in df_rows.columns
    y_col = "efficiency" if use_efficiency else "current_cr"
    y_label = "Efficiency" if use_efficiency else "Compression Ratio (CR)"

    at_step = df_rows[df_rows["step"] == step]
    if at_step.empty:
        return go.Figure().update_layout(
            title=f"No data at step {step}",
            template="plotly_white", height=380,
        )

    fig = go.Figure()

    for policy in sorted(at_step["Algo"].unique()):
        p = at_step[at_step["Algo"] == policy].sort_values("step_loss")
        color = get_policy_color(policy)
        labels = p.apply(
            lambda r: (
                f"<b>{policy.upper()}</b> — Seed {r['Seed']}<br>"
                f"Loss: {r['step_loss']:.5f}<br>"
                + (f"Efficiency: {r[y_col]:.3f}" if use_efficiency else f"CR: {r[y_col]:.3f}")
            ), axis=1
        )
        fig.add_trace(go.Scatter(
            x=p["step_loss"],
            y=p[y_col],
            mode="lines+markers",
            line=dict(color=color, width=1.5, dash="dot"),
            marker=dict(color=color, size=10, line=dict(color="white", width=1)),
            name=policy.upper(),
            text=labels.tolist(),
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.4)
    fig.add_hline(
        y=1, line_dash="dash", line_color="gray", opacity=0.6,
        annotation_text="y = 1", annotation_position="right",
        annotation_font=dict(color="gray", size=11),
    )

    fig.update_layout(
        title=dict(text=f"Pareto view at step {step}: Loss vs {y_metric}", font=dict(size=13)),
        xaxis=dict(title="Step loss", autorange="reversed" if use_efficiency else True),
        yaxis_title=y_label,
        height=380,
        template="plotly_white",
        legend=dict(orientation="v", x=1.02, y=1, xanchor="left"),
        margin=dict(l=40, r=160, t=50, b=50),
    )
    return fig
