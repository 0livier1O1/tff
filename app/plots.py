"""
Reusable Plotly plot functions for the BOSS dashboard.
Each function accepts a df_rows DataFrame (standard traces format) and
returns a plotly.graph_objects.Figure.

Expected columns: step, cur_loss, chosen_loss, current_cr, selected_arm,
  oracle_best_arm, oracle_arm_rank, oracle_best_loss, regret, cum_regret,
  arm_match, pick_time_s, decomp_time_s, gp_fit_time_s, oracle_time_s, step_time_s, Policy, Seed.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

POLICY_COLORS = {
    "mabss-greedy": "#4E79A7",
    "mabss-ucb": "#E15759",
    "mabss-exp3": "#59A14F",
    "mabss-exp4": "#F28E2B",
    "boss-ei": "#9467BD",
    "boss-ucb": "#8C564B",
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


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _mean_std_over_seeds(df_rows, y_col, x_col="step"):
    """Return per-policy (x, mean, lower, upper) aggregated over seeds."""
    result = {}
    for policy in sorted(df_rows["Policy"].unique()):
        pol = df_rows[df_rows["Policy"] == policy].copy().sort_values(["Seed", "step"])
        if x_col == "cum_time_s":
            pol["cum_time_s"] = pol.groupby("Seed")["step_time_s"].cumsum()
        gb = pol.groupby("step")
        mean = gb[y_col].mean()
        std = gb[y_col].std().fillna(0)
        x = gb[x_col].mean() if x_col != "step" else mean.index
        result[policy] = (
            x.values,
            mean.values,
            (mean - std).values,
            (mean + std).values,
        )
    return result


def _effective_step_time(df: pd.DataFrame, policy: str) -> pd.Series:
    """
    Algorithmic wall-clock time per step, excluding oracle evaluation time
    for non-greedy policies (oracle is only bookkeeping overhead there).
    For greedy, oracle IS the selection cost, so it is included.
    """
    is_greedy = policy.lower().replace("mabss-", "") == "greedy"
    base_cols = ["decomp_time_s", "gp_fit_time_s", "pick_time_s"]
    # Fall back to step_time_s if new columns aren't present (legacy traces)
    if "decomp_time_s" not in df.columns:
        if is_greedy:
            return df["step_time_s"]
        return df["step_time_s"] - df.get("oracle_time_s", 0)
    t = df[base_cols].sum(axis=1)
    if is_greedy:
        t = t + df["oracle_time_s"]
    return t


def _interpolate_on_time_grid(
    df_rows: pd.DataFrame, policy: str, y_col: str, n_points: int = 300
):
    """
    Zero-order hold interpolation of any step-wise metric onto a shared time
    grid across seeds. Between steps the value is constant (nothing changed),
    so we hold the last observation until the next step fires.
    Grid runs 0 → earliest seed finish to avoid extrapolation.
    """
    pol = df_rows[df_rows["Policy"] == policy].copy().sort_values(["Seed", "step"])
    pol["_eff_t"] = _effective_step_time(pol, policy)
    pol["cum_time_s"] = pol.groupby("Seed")["_eff_t"].cumsum()
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
    std = curves.std(axis=0)
    return grid, mean, mean - std, mean + std


# ---------------------------------------------------------------------------
# Moved from dashboard.py — these are the canonical dashboard plot functions
# ---------------------------------------------------------------------------


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
            "Loss vs. Step (± 1 SD)",
            "Regret vs. Step (± 1 SD)",
            "Loss vs. Cumulative Runtime (± 1 SD)",
            "Regret vs. Cumulative Runtime (± 1 SD)",
        ),
        vertical_spacing=0.18,
    )

    for policy in df_rows["Policy"].unique():
        color = get_policy_color(policy)
        rgb = _rgba(color, 0.2)
        sub = df_rows[df_rows["Policy"] == policy]
        gb = sub.groupby("step")
        steps = list(gb.groups.keys())

        mean_loss = gb["chosen_loss"].mean()
        std_loss = gb["chosen_loss"].std().fillna(0)
        mean_regret = gb["cum_regret"].mean()
        std_regret = gb["cum_regret"].std().fillna(0)

        t_loss, m_loss, lo_loss, hi_loss = _interpolate_on_time_grid(
            df_rows, policy, "chosen_loss", n_points
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
        _band(steps, mean_loss - std_loss, mean_loss + std_loss, 1, 1)
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

        _band(steps, mean_regret - std_regret, mean_regret + std_regret, 1, 2)
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


def plot_arm_trace(sub: pd.DataFrame, pol_name: str, color: str) -> go.Figure:
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
            text=f"{pol_name.upper()} — hit rate={hit_rate:.2f}, unique arms={unique_arms}",
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
    for policy in sorted(seed_df["Policy"].unique()):
        c = get_policy_color(policy)
        pol = seed_df[seed_df["Policy"] == policy].sort_values("step").copy()
        pol["cum_time_s"] = _effective_step_time(pol, policy).cumsum()
        fig.add_trace(
            go.Scatter(
                x=pol["cum_time_s"].values,
                y=pol["cur_loss"].values,
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
    for policy in sorted(df_rows["Policy"].unique()):
        x, mean, lo, hi = _interpolate_on_time_grid(
            df_rows, policy, "cur_loss", n_points
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
    for policy in sorted(df_rows["Policy"].unique()):
        c = get_policy_color(policy)
        pol = df_rows[df_rows["Policy"] == policy].copy().sort_values(["Seed", "step"])
        pol["_eff_t"] = _effective_step_time(pol, policy)
        pol["cum_time_s"] = pol.groupby("Seed")["_eff_t"].cumsum()
        seeds = sorted(pol["Seed"].unique())
        n_seeds = len(seeds)
        for i, seed in enumerate(seeds):
            s = pol[pol["Seed"] == seed].sort_values("cum_time_s")
            alpha = 1.0 - 0.5 * (i / max(n_seeds - 1, 1))
            fig.add_trace(
                go.Scatter(
                    x=s["cum_time_s"].values,
                    y=s["cur_loss"].values,
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
    policies = sorted(df_rows["Policy"].unique())
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
        pol_df = df_rows[df_rows["Policy"] == policy].sort_values(["Seed", "step"])
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
    for policy in sorted(df_rows["Policy"].unique()):
        c = get_policy_color(policy)
        last = (
            df_rows[df_rows["Policy"] == policy]
            .sort_values("step")
            .groupby("Seed")
            .last()
            .reset_index()
        )
        fig.add_trace(
            go.Scatter(
                x=last["current_cr"],
                y=last["cur_loss"],
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


def plot_decomp_curves(decomp_data: list[dict], pol_name: str, color: str) -> go.Figure:
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

    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)

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
            line=dict(color=f"rgba({r},{g},{b},0.4)", width=1, dash="dash"),
        ))
        annotations.append(dict(
            x=x0, y=1, yref="paper",
            text=f"s{step}<br>a{arm}",
            showarrow=False,
            font=dict(size=9, color=f"rgba({r},{g},{b},0.8)"),
            xanchor="left",
            yanchor="top",
        ))

    fig.update_layout(
        title=dict(text=f"{pol_name.upper()} — decomposition convergence", font=dict(size=12)),
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
    """Scatter: cumulative runtime vs CR or Efficiency at first step where chosen_loss <= threshold.

    Each point is one (policy, seed) pair. Colored by policy.
    Points where the threshold is never reached are omitted.
    """
    if "chosen_loss" not in df_rows.columns or "step_time_s" not in df_rows.columns:
        return go.Figure()

    use_efficiency = y_metric == "Efficiency" and "efficiency" in df_rows.columns
    y_col = "efficiency" if use_efficiency else "current_cr"
    y_label = "Efficiency (CR / target CR)" if use_efficiency else "Compression Ratio (CR)"

    # Cumulative runtime per (Policy, Seed) in step order
    df = df_rows.sort_values(["Policy", "Seed", "step"]).copy()
    df["cum_time"] = df.groupby(["Policy", "Seed"])["step_time_s"].cumsum()

    keep_cols = ["Policy", "Seed", "step", "chosen_loss", "current_cr", "cum_time"]
    if use_efficiency:
        keep_cols.append("efficiency")

    # First row per (Policy, Seed) where loss hits threshold
    hits = (
        df[df["chosen_loss"] <= threshold]
        .groupby(["Policy", "Seed"], sort=False)
        .first()
        .reset_index()[keep_cols]
    )

    if hits.empty:
        return go.Figure().update_layout(
            title=f"No policy reached loss ≤ {threshold}",
            template="plotly_white", height=420,
        )

    policies = sorted(hits["Policy"].unique())

    hits["label"] = hits.apply(
        lambda r: (
            f"<b>{r['Policy'].upper()}</b> — Seed {r['Seed']}<br>"
            f"Runtime: {r['cum_time']:.1f}s<br>"
            f"CR: {r['current_cr']:.3f}"
            + (f"<br>Efficiency: {r['efficiency']:.3f}" if use_efficiency else "")
            + f"<br>Step: {int(r['step'])}<br>Loss: {r['chosen_loss']:.5f}"
        ), axis=1
    )

    fig = go.Figure()

    for policy in policies:
        p = hits[hits["Policy"] == policy]
        sizes = 9 + (p["chosen_loss"] / threshold) * 10
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
            text=f"Time-to-threshold (loss ≤ {threshold}): Runtime vs {y_metric}",
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
