"""
render.py — all Streamlit rendering for the BOSS dashboard.

Three public functions:
  render_job_status_panel(ROOT)        always-visible active-run tracker
  render_load_mode(ROOT)               Load Past Artifact sidebar + data loading
  render_results(...)                  unified results visualization
"""
from __future__ import annotations

import json
from datetime import datetime as _dt, timedelta as _td
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Active-run status panel
# ---------------------------------------------------------------------------

def render_job_status_panel(ROOT: Path) -> None:
    """Display and update the active-run tracker. Reads/writes session state."""
    import time as _time
    from app.utils import _script_alive, _job_status

    active_runs = st.session_state.get("active_runs", [])
    if not active_runs:
        return

    _hdr, _btn = st.columns([5, 1])
    _hdr.markdown("#### Active Runs")
    if _btn.button("Refresh", use_container_width=True):
        st.rerun()

    def _fmt_ts(ts):
        return _dt.fromtimestamp(ts).strftime("%H:%M:%S") if ts else ""

    def _fmt_dur(start_ts, end_ts=None):
        if not start_ts:
            return ""
        secs = int((end_ts or _time.time()) - start_ts)
        return str(_td(seconds=secs))

    still_active = []
    for rec in active_runs:
        rname = rec["run_name"]
        out_dir = ROOT / "artifacts" / rname
        alive = _script_alive(Path(rec["pid_file"]))
        st.markdown(f"**`{rname}`**")

        cfg_data = {}
        cfg_file = out_dir / "config.json"
        if cfg_file.exists():
            try:
                with open(cfg_file) as f:
                    cfg_data = json.load(f)
            except Exception:
                pass

        submitted_at = rec.get("submitted_at")
        rows, all_done = [], True

        for job in rec["jobs"]:
            status, step = _job_status(job, alive)
            if status != "Done":
                all_done = False

            algo_dir = Path(job.get("algo_dir", job.get("pol_dir", "")))
            pf = algo_dir / "progress.json"
            done_f = algo_dir / ".done"

            started_at = None
            if pf.exists():
                try:
                    started_at = json.loads(pf.read_text()).get("started_at")
                except Exception:
                    pass
            completed_at = done_f.stat().st_mtime if done_f.exists() else None

            _algo = job["algo"]
            if _algo.startswith("mabss-"):
                _budget = cfg_data.get("mabss_budget", "-")
                _epochs = cfg_data.get("mabss_decomp_epochs", "-")
                _max_rank = cfg_data.get("mabss_max_rank", "-")
            elif _algo.startswith("boss-"):
                _budget = cfg_data.get("boss_budget", "-")
                _epochs = cfg_data.get("boss_decomp_epochs", "-")
                _max_rank = cfg_data.get("boss_max_bond", "-")
            elif _algo == "tnale":
                _budget = cfg_data.get("tnale_budget", "-")
                _epochs = cfg_data.get("tnale_decomp_epochs", "-")
                _max_rank = cfg_data.get("tnale_max_rank", "-")
            else:
                _budget = "-"
                _epochs = "-"
                _max_rank = "-"
            rows.append({
                "Seed":      job["seed"],
                "Algo":      _algo,
                "Status":    status,
                "Step":      step,
                "N":         cfg_data.get("n_cores", "-"),
                "Budget":    _budget,
                "Epochs":    _epochs,
                "MaxRank":   _max_rank,
                "Submitted": _fmt_ts(submitted_at),
                "Started":   _fmt_ts(started_at),
                "Duration":  _fmt_dur(started_at, completed_at),
                "Completed": _fmt_ts(completed_at),
            })

        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        if all_done:
            (out_dir / "session_state.json").unlink(missing_ok=True)
            st.sidebar.success(f"`{rname}` complete — load it via Load Past Artifact.")
        else:
            still_active.append(rec)

    st.session_state["active_runs"] = still_active


# ---------------------------------------------------------------------------
# Load Past Artifact
# ---------------------------------------------------------------------------

def render_load_mode(ROOT: Path):
    """
    Render the Load Past Artifact sidebar widgets and load data from disk.

    Returns (data_ready, df_mabss, df_boss, df_tnale, summaries, decomp_dict, df_summary, out_dir).
    When data_ready is False all other values are None / empty.
    """
    from app.sidebar import DEFAULT_PARAMS
    from app.utils import _load_artifact, _artifact_fully_done

    _empty = (False, None, None, None, None, {}, {}, None, None)

    st.sidebar.markdown("### Historical Archives")
    artifact_dir = ROOT / "artifacts"

    if not artifact_dir.exists():
        st.sidebar.error("No artifacts directory found. Run an evaluation first.")
        st.stop()

    past_runs = sorted(
        [d.name for d in artifact_dir.iterdir() if d.is_dir() and _artifact_fully_done(d)],
        reverse=True,
    )
    if not past_runs:
        st.sidebar.warning("No complete artifacts found.")
        st.stop()

    MODE_SUMMARY = "-- View Global Summary Table --"
    selected_run = st.sidebar.selectbox("Select Cached Run", [MODE_SUMMARY] + past_runs)

    if selected_run == MODE_SUMMARY:
        st.session_state["loaded_run"] = ""
        _render_global_summary(artifact_dir, past_runs, DEFAULT_PARAMS)
        return _empty

    if not selected_run:
        return _empty

    out_dir = artifact_dir / selected_run
    st.sidebar.success(f"Viewing Historical Artifact: `{selected_run}`")
    try:
        df_mabss, df_boss, df_tnale, summaries, decomp_dict, pol_diagnostics_dict = _load_artifact(out_dir)
        if df_mabss is None and df_boss is None and df_tnale is None:
            st.error(
                "Artifact contains no valid multi-seed environments. "
                "Legacy runs without seeded geometries are strictly deprecated."
            )
            st.stop()

        st.session_state["loaded_run"] = selected_run

        with open(out_dir / "config.json") as f:
            cfg_json = json.load(f)
        st.sidebar.markdown("### Static Configuration")
        c1, c2 = st.sidebar.columns(2)
        c1.metric("Cores ($N$)", cfg_json.get("n_cores", "-"))
        c2.metric("Max Search Rank", cfg_json.get("max_edge_rank", "-"))
        b1, b2 = st.sidebar.columns(2)
        b1.metric("Steps Budget", cfg_json.get("budget", "-"))
        b2.metric("Decomp Epochs", f"{cfg_json.get('mabss_decomp_epochs', '-')}/{cfg_json.get('boss_decomp_epochs', '-')}")
        with st.sidebar.expander("Underlying Hyperparameters", expanded=False):
            st.json(cfg_json)

        df_summary = pd.DataFrame(summaries)
        return True, df_mabss, df_boss, df_tnale, summaries, decomp_dict, pol_diagnostics_dict, df_summary, out_dir

    except Exception:
        st.error("Failed to load artifact due to filesystem drift.")
        st.stop()


def _render_global_summary(artifact_dir: Path, past_runs: list[str], DEFAULT_PARAMS: dict) -> None:
    st.markdown("## Global Artifact Aggregation")
    all_configs = []
    for run in past_runs:
        cfile = artifact_dir / run / "config.json"
        if cfile.exists():
            try:
                with open(cfile) as f:
                    cfg = json.load(f)
                    cfg["run_name"] = run
                    all_configs.append(cfg)
            except Exception:
                pass

    if not all_configs:
        return

    df_hist = pd.DataFrame(all_configs)
    cols = ["run_name"] + [c for c in df_hist.columns if c != "run_name" and c in DEFAULT_PARAMS]
    df_hist = df_hist[[c for c in cols if c in df_hist.columns]]
    df_hist.set_index("run_name", inplace=True)

    def _highlight(val, col_name):
        if col_name in DEFAULT_PARAMS and str(val) != str(DEFAULT_PARAMS[col_name]):
            return "background-color: lightcoral; color: white;"
        return ""

    styled = df_hist.style.apply(lambda col: [_highlight(v, col.name) for v in col], axis=0)
    st.markdown(
        "Metrics highlighted in **red** mathematically deviate from canonical backend defaults. "
        "Choose a specific run from the sidebar to drill into algorithmic trace arrays."
    )
    st.dataframe(styled, use_container_width=True, height=600)


# ---------------------------------------------------------------------------
# Unified results renderer
# ---------------------------------------------------------------------------

def render_results(
    df_mabss: pd.DataFrame,
    df_boss: pd.DataFrame,
    df_tnale: pd.DataFrame,
    summaries: list[dict],
    decomp_dict: dict,
    pol_diagnostics_dict: dict,
    df_summary: pd.DataFrame,
    out_dir: Path,
    ROOT: Path,
) -> None:
    """Render the full results view: macro plots, per-seed drill-downs, export."""
    from app.plots import (
        plot_boss_objective,
        plot_loss_and_regret,
        plot_arm_trace,
        plot_loss_vs_runtime_seed,
        plot_step_time_breakdown,
        plot_decomp_curves,
        plot_time_to_threshold,
        plot_pareto_at_step,
    )
    from app.utils import get_policy_color, _cr_from_adj

    frames = [df for df in (df_mabss, df_boss, df_tnale) if df is not None and not df.empty]
    df_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    st.markdown("## Global Performance Overview")

    _has_mabss = df_mabss is not None and not df_mabss.empty
    _has_boss = df_boss is not None and not df_boss.empty

    if _has_mabss:
        st.markdown("#### MABSS — Loss & Regret")
        _mabss_steps = sorted(df_mabss["step"].dropna().unique().astype(int))
        _mabss_slider_col, _ = st.columns([1, 3])
        with _mabss_slider_col:
            _mabss_max_step = st.select_slider(
                "Max step shown (MABSS)",
                options=_mabss_steps,
                value=_mabss_steps[-1],
                key="mabss_step_crop",
            )
        st.plotly_chart(
            plot_loss_and_regret(df_mabss[df_mabss["step"] <= _mabss_max_step]),
            use_container_width=True,
            key="loss_and_regret_mabss",
        )

    if _has_boss:
        st.markdown("#### BOSS — Objective Convergence")
        _boss_steps = sorted(df_boss["step"].dropna().unique().astype(int))
        _boss_slider_col, _ = st.columns([1, 3])
        with _boss_slider_col:
            _boss_max_step = st.select_slider(
                "Max step shown (BOSS)",
                options=_boss_steps,
                value=_boss_steps[-1],
                key="boss_step_crop",
            )
        st.plotly_chart(
            plot_boss_objective(df_boss[df_boss["step"] <= _boss_max_step]),
            use_container_width=True,
            key="boss_objective_global",
        )

    if not df_rows.empty:
        st.markdown("#### Performance vs Runtime")
        _ctrl_col, _chart_col = st.columns([1, 5])
        with _ctrl_col:
            _threshold = st.number_input(
                "Loss threshold",
                min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.3f",
                key="ttt_threshold",
            )
            _has_eff = "efficiency" in df_rows.columns
            _y_metric = st.selectbox(
                "Y-axis",
                options=["CR", "Efficiency"] if _has_eff else ["CR"],
                key="ttt_y_metric",
            )
        with _chart_col:
            st.plotly_chart(
                plot_time_to_threshold(df_rows, threshold=_threshold, y_metric=_y_metric),
                use_container_width=True,
                key="time_to_threshold_global",
            )

        st.markdown("#### Pareto Curves")
        _pareto_ctrl, _pareto_chart = st.columns([1, 5])
        with _pareto_ctrl:
            _available_steps = sorted(df_rows["step"].dropna().unique().astype(int))
            _pareto_step = st.select_slider(
                "Step",
                options=_available_steps,
                value=_available_steps[-1],
                key="pareto_step",
            )
            _pareto_y_metric = st.selectbox(
                "Y-axis",
                options=["CR", "Efficiency"] if _has_eff else ["CR"],
                key="pareto_y_metric",
            )
        with _pareto_chart:
            st.plotly_chart(
                plot_pareto_at_step(df_rows, step=_pareto_step, y_metric=_pareto_y_metric),
                use_container_width=True,
                key="pareto_global",
            )

    st.divider()
    st.markdown("## Seed-Specific Analysis Maps")

    for seed in sorted(df_rows["Seed"].unique()):
        with st.expander(f"Seed {seed} — Execution Trace", expanded=False):
            seed_df = df_rows[df_rows["Seed"] == seed]
            if seed_df.empty:
                continue

            seed_summaries = [s for s in summaries if s.get("Seed") == seed]
            algo_names = [s["algo"] for s in seed_summaries]
            s_dir = out_dir / f"seed_{seed}" if (out_dir / f"seed_{seed}").exists() else out_dir

            # Summary table
            st.markdown("#### Summary")
            if seed_summaries:
                _render_summary_table(seed_df, seed_summaries, s_dir, get_policy_color, _cr_from_adj)
                _render_generating_rse(s_dir, seed)

            st.divider()

            # Topology images
            st.markdown("#### Visualizing Tensor and Topology")
            _render_topology(s_dir, algo_names)

            st.divider()

            # Timing charts
            st.markdown("#### Computational Cost")
            _col_rt, _col_bd = st.columns(2)
            with _col_rt:
                st.plotly_chart(
                    plot_loss_vs_runtime_seed(seed_df),
                    use_container_width=True, key=f"loss_vs_runtime_{seed}",
                )
            with _col_bd:
                st.plotly_chart(
                    plot_step_time_breakdown(seed_df),
                    use_container_width=True, key=f"step_time_breakdown_{seed}",
                )

            # Trace vectors
            st.markdown("#### Mathematical Trace Vectors")
            for s in seed_summaries:
                algo_name = s["algo"]
                c = get_policy_color(algo_name)
                sub = seed_df[seed_df["Algo"] == algo_name]
                if sub.empty:
                    continue
                _decomp_data = decomp_dict.get((seed, algo_name), [])
                if _decomp_data:
                    _col_arm, _col_decomp = st.columns(2)
                    with _col_arm:
                        st.plotly_chart(
                            plot_arm_trace(sub, algo_name, c),
                            use_container_width=True, key=f"arm_trace_{seed}_{algo_name}",
                        )
                    with _col_decomp:
                        st.plotly_chart(
                            plot_decomp_curves(_decomp_data, algo_name, c),
                            use_container_width=True, key=f"decomp_curves_{seed}_{algo_name}",
                        )
                else:
                    st.plotly_chart(
                        plot_arm_trace(sub, algo_name, c),
                        use_container_width=True, key=f"arm_trace_{seed}_{algo_name}",
                    )

    st.divider()
    st.markdown("### Export Artifacts")
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            label="Download Trajectory Data (CSV)",
            data=df_rows.to_csv(index=False).encode("utf-8"),
            file_name="traces_global.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            label="Download Metric Summary (JSON)",
            data=df_summary.to_json(orient="records").encode("utf-8"),
            file_name="summary_global.json",
            mime="application/json",
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Private render helpers
# ---------------------------------------------------------------------------

def _render_summary_table(seed_df, seed_summaries, s_dir, get_policy_color, _cr_from_adj):
    import numpy as _np

    _summary_rows = []
    for s in seed_summaries:
        _summary_rows.append({
            "algo": s.get("algo"),
            "final_step_loss": s.get("final_step_loss", s.get("final_loss_after_move")),
            "final_cr": s.get("final_cr"),
            "cumulative_regret": s.get("cumulative_regret"),
            "oracle_hit_rate": s.get("oracle_hit_rate"),
            "unique_arms": s.get("unique_arms"),
            "arm_entropy_norm": s.get("arm_entropy_norm"),
            "steps": s.get("steps"),
            "budget": s.get("budget"),
        })
    _sum_keys = [
        "algo", "final_step_loss", "final_cr", "cumulative_regret",
        "oracle_hit_rate", "unique_arms", "arm_entropy_norm", "steps", "budget",
    ]
    _sum_labels = [
        "Algo", "Final Loss", "Final CR", "Cum. Regret",
        "Oracle Hit Rate", "Unique Arms", "Arm Entropy", "Steps", "Budget",
    ]
    df_sum = pd.DataFrame([{k: s.get(k) for k in _sum_keys} for s in _summary_rows])
    df_sum.columns = _sum_labels

    _total_time = (
        seed_df.groupby("Algo")["step_time_s"].sum().round(1)
        if "step_time_s" in seed_df.columns else pd.Series(dtype=float)
    )
    df_sum["Total Time (s)"] = df_sum["Algo"].map(_total_time)

    _target_npz = s_dir / "target_tensor.npz"
    if _target_npz.exists():
        _shape = _np.load(_target_npz)["data"].shape
        df_sum["Shape"] = "×".join(str(d) for d in _shape)

    if "efficiency" in seed_df.columns:
        _last_eff = seed_df.groupby("Algo")["efficiency"].last()
        df_sum["Efficiency"] = df_sum["Algo"].map(_last_eff).round(3)

    df_sum["Final Loss"] = df_sum["Final Loss"].round(4)
    df_sum["Final CR"] = df_sum["Final CR"].round(3)
    df_sum["Cum. Regret"] = df_sum["Cum. Regret"].round(4)
    df_sum["Oracle Hit Rate"] = df_sum["Oracle Hit Rate"].round(3)
    df_sum["Arm Entropy"] = df_sum["Arm Entropy"].round(3)

    def _style_row(row):
        c = get_policy_color(row["Algo"])
        styles = [f"background-color: {c}15"] * len(row)
        styles[0] = f"background-color: {c}; color: white; font-weight: bold; padding-left: 10px;"
        return styles

    st.dataframe(df_sum.style.apply(_style_row, axis=1), hide_index=True, use_container_width=True)


def _render_generating_rse(s_dir: Path, seed: int) -> None:
    import plotly.graph_objects as _go

    gen_path = s_dir / "generating_rse.json"
    if not gen_path.exists():
        return
    with open(gen_path) as f:
        gen = json.load(f)
    losses = gen.get("losses", [])
    if not losses:
        return

    fig = _go.Figure()
    fig.add_trace(_go.Scatter(
        x=list(range(1, len(losses) + 1)),
        y=losses,
        mode="lines",
        line=dict(color="#444444", width=1.5),
        hovertemplate="Epoch %{x}<br>RSE %{y:.5f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"Generating structure decomposition — best RSE: {gen['rse']:.5f}  CR: {gen['cr']:.3f}",
            font=dict(size=12),
        ),
        xaxis_title="Epoch",
        yaxis_title="RSE",
        yaxis_type="log",
        height=260,
        template="plotly_white",
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"gen_rse_{seed}")


def _render_topology(s_dir: Path, algo_names: list[str]) -> None:
    target_img_path = _find_file(s_dir, algo_names, "target_image.png")
    target_graph_path = _find_file(s_dir, algo_names, "target_graph.png")

    has_img = target_img_path is not None
    has_graph = target_graph_path is not None

    if has_img and algo_names:
        n_cols = len(algo_names) + 1
        row1 = st.columns(n_cols)
        with row1[0]:
            st.image(str(target_img_path), caption="Target", use_container_width=True)
        for i, algo_name in enumerate(algo_names):
            p_subdir = _algo_subdir(s_dir, algo_name)
            p_img = p_subdir / "reconstruction.png"
            if not p_img.exists():
                p_img = s_dir / f"reconstruction_{algo_name.replace('-', '_')}.png"
            with row1[i + 1]:
                if p_img.exists():
                    st.image(str(p_img), caption=algo_name.upper(), use_container_width=True)
                else:
                    st.info(f"No {algo_name} recon")

        st.markdown("<br>", unsafe_allow_html=True)
        row2 = st.columns(n_cols)
        for i, algo_name in enumerate(algo_names):
            p_sub = _algo_subdir(s_dir, algo_name)
            p_graph = _algo_graph(p_sub, algo_name)
            with row2[i + 1]:
                if p_graph:
                    st.image(str(p_graph), caption=algo_name.upper(), use_container_width=True)
    else:
        if has_graph:
            _, t_c2, _ = st.columns([1, 2, 1])
            with t_c2:
                st.image(str(target_graph_path), caption="Target Ground Truth Structure", use_container_width=True)
        if algo_names:
            st.markdown("<br>", unsafe_allow_html=True)
            p_cols = st.columns(len(algo_names))
            for i, algo_name in enumerate(algo_names):
                p_sub = _algo_subdir(s_dir, algo_name)
                p_graph = _algo_graph(p_sub, algo_name, fallback=s_dir)
                with p_cols[i]:
                    if p_graph:
                        st.image(str(p_graph), caption=f"Found: {algo_name.upper()}", use_container_width=True)
                    else:
                        st.info(f"No {algo_name} structure")


def _find_file(s_dir: Path, algo_names: list[str], filename: str) -> Path | None:
    """Look for filename in s_dir first, then in policy subdirs."""
    direct = s_dir / filename
    if direct.exists():
        return direct
    for p_name in algo_names:
        p_base = p_name.replace("-", "_")
        for pfx in ["", "mabss_", "boss_"]:
            cand = s_dir / f"{pfx}{p_base}" / filename
            if cand.exists():
                return cand
    return None


def _algo_subdir(s_dir: Path, algo_name: str) -> Path:
    p_base = algo_name.replace("-", "_")
    p_sub = s_dir / p_base
    if not p_sub.exists():
        for pfx in ["mabss_", "boss_"]:
            candidate = s_dir / f"{pfx}{p_base}"
            if candidate.exists():
                return candidate
    return p_sub


def _algo_graph(p_sub: Path, algo_name: str, fallback: Path | None = None) -> Path | None:
    p_base = algo_name.replace("-", "_")
    short_p = algo_name.split("-")[-1]
    candidates = [
        p_sub / f"tn_graph_{algo_name}.png",
        p_sub / f"tn_graph_{short_p}.png",
        p_sub / f"tn_graph_{p_base}.png",
    ]
    if fallback:
        candidates += [
            fallback / f"tn_graph_{short_p}.png",
            fallback / f"tn_graph_{p_base}.png",
        ]
    for c in candidates:
        if c.exists():
            return c
    return None
