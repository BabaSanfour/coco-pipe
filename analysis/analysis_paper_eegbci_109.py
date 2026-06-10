# ruff: noqa: E501
"""Paper-grade PCA neural-trajectory analysis on PhysioNet EEGBCI (109 subjects).

Headless counterpart to the main tutorial notebook
(`tutorials/tutorial_pca_trajectories_eegbci.ipynb`). Runs the full 10-step
PCA-trajectory workflow at paper scale and produces a standalone HTML report
that mirrors the same steps as the tutorial notebook.

Outputs (under ``--output``)::

    paper_figure.svg / .html       — 5-panel manuscript composite
    report.html                    — 10-step interactive companion report
    artifacts/                     — per-subject reducers + container + CSVs

Usage
-----
::

    python analysis/analysis_paper_eegbci_109.py \\
        --subjects 1 2 3 ... 109 \\
        --output outputs/paper/eegbci_109 \\
        --n-perm 1000

By default runs on all 109 subjects with ``n_perm=1000`` for the
permutation null. Pass ``--subjects 1 2 3`` to dry-run on a small
slice (matches the tutorial mode).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Allow `python analysis/analysis_paper_eegbci_109.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy.signal import find_peaks  # noqa: E402

from coco_pipe.dim_reduction import (  # noqa: E402
    DimReduction,
    paired_condition_stats,
    trajectory_dispersion,
    trajectory_jerk,
    trajectory_path_length,
    trajectory_speed,
    trajectory_turning_angle,
)
from coco_pipe.io.dataset import BIDSDataset  # noqa: E402
from coco_pipe.report import PlotlyElement, Report, Section  # noqa: E402
from coco_pipe.report.elements import (  # noqa: E402
    AccordionElement,
    InteractiveTableElement,
    MarkdownElement,
)
from coco_pipe.viz.interactive import (  # noqa: E402
    plot_bar,
    plot_distribution_groups,
    plot_eigenvalues,
    plot_heatmap,
    plot_null_interval_summary,
    plot_shepard_diagram,
    plot_trajectory,
    plot_trajectory_metric_series,
    plot_trajectory_separation,
)
from tutorials._helpers import (  # noqa: E402
    compute_per_condition_scalars,
    compute_per_trial_scalars,
    compute_separation_pair_scalars,
    compute_separation_timecourses,
    permutation_null_separation_auc,
    save_artifacts,
    setup_data_bids,
)

# ---------------------------------------------------------------------------
# EEGBCI condition metadata (mirrors tutorials/tutorial_pca_trajectories_eegbci.ipynb)
# ---------------------------------------------------------------------------
_CB = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#F0E442",
    "#56B4E9",
    "#E69F00",
    "#000000",
)

LABEL_NAMES: dict[int, str] = {
    3: "Left Hand (Exec)",
    4: "Right Hand (Exec)",
    5: "Left Hand (Imag)",
    6: "Right Hand (Imag)",
    7: "Hands (Exec)",
    8: "Feet (Exec)",
    9: "Hands (Imag)",
    10: "Feet (Imag)",
}
CONDITION_COLORS: dict[int, str] = {
    3: _CB[0],
    4: _CB[1],
    7: _CB[2],
    8: _CB[3],
    5: _CB[4],
    6: _CB[5],
    9: _CB[6],
    10: _CB[7],
}
DEFAULT_CONDITIONS: tuple[int, ...] = (3, 4, 7, 8)

# BIDS trial_type values written by setup_data_bids (used as event_id keys)
BIDS_NAMES: dict[int, str] = {
    3: "left_hand_exec",
    4: "right_hand_exec",
    5: "left_hand_imag",
    6: "right_hand_imag",
    7: "hands_exec",
    8: "feet_exec",
    9: "hands_imag",
    10: "feet_imag",
}


def _resolve_subjects(spec: list[int] | None) -> list[int]:
    if spec is None or not spec:
        return list(range(1, 110))
    return [int(s) for s in spec]


def run_paper_analysis(
    subjects: list[int] | None,
    output: Path | str,
    bids_root: Path | str = "PhysioNet_EEGBCI/BIDS",
    n_perm: int = 1000,
    n_components: int = 10,
    analysis_window: tuple[float, float] = (0.0, 1.5),
    random_state: int = 42,
) -> dict:
    """Run the 10-step PCA-trajectory workflow at paper scale.

    See module docstring for output layout.
    """
    rng = np.random.default_rng(random_state)
    subject_list = _resolve_subjects(subjects)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    artifacts_dir = out / "artifacts"

    print("=== EEGBCI paper analysis ===")
    print(
        f"subjects: {subject_list[0]}..{subject_list[-1]} ({len(subject_list)} total)"
    )
    print(f"output:   {out}")

    # ------------------------------------------------------------------ Steps 1–3
    # Step 1: Representation — sensor space (64 EEG channels), 4 motor-imagery conditions
    # Step 2: Preprocess — 0.5–40 Hz bandpass + 50/60 Hz notch + ICA blink removal
    setup_data_bids(subjects=subject_list, runs=list(range(3, 15)), root=bids_root)

    _event_id = {BIDS_NAMES[k]: k for k in DEFAULT_CONDITIONS}
    container = BIDSDataset(
        root=str(bids_root),
        task="motorimagery",
        datatype="eeg",
        mode="epochs",
        event_id=_event_id,
        runs=[f"{r:02d}" for r in range(3, 15)],
        tmin=analysis_window[0] - 1.0,
        tmax=analysis_window[1] + 1.0,
        baseline=(-1.0, 0.0),
    ).load()

    times_full = np.asarray(container.coords["time"])
    subject_ids = np.asarray(container.coords["subject"])

    # Step 3: Reshape — crop to analysis window, baseline + z-score, then stack
    time_mask = (times_full >= analysis_window[0] - 1e-8) & (
        times_full <= analysis_window[1] + 1e-8
    )
    container_win = container.isel(time=time_mask)
    times = np.asarray(container_win.coords["time"])

    container_z = container_win.baseline_correction(dim="time").zscore(dim="obs")
    container_pooled = container_z.stack(dims=("obs", "time"), new_dim="obs")
    print(f"Pooled stack shape: {container_pooled.X.shape}")

    # ------------------------------------------------------------------ Step 4a / 4b
    print("Fitting shared PCA …")
    shared_reducer = DimReduction(
        method="PCA", n_components=n_components, random_state=random_state
    )
    shared_scores = shared_reducer.fit_transform(container_pooled.X)
    evr = np.asarray(shared_reducer.get_diagnostics().get("explained_variance_ratio_"))

    print("Fitting per-subject PCA …")
    per_subject_reducers: dict = {}
    for sub_id in np.unique(subject_ids):
        sub_mask = subject_ids == sub_id
        sub_stack = container_z.isel(obs=sub_mask).stack(
            dims=("obs", "time"), new_dim="obs"
        )
        red = DimReduction(
            method="PCA", n_components=n_components, random_state=random_state
        )
        red.fit_transform(sub_stack.X)
        per_subject_reducers[sub_id] = red

    # ------------------------------------------------------------------ Step 5: scree
    pr = float((evr.sum() ** 2) / (evr**2).sum())
    print(f"Cumulative variance @ 3 PCs: {float(np.sum(evr[:3])):.2%}  |  PR: {pr:.2f}")
    fig_scree = plot_eigenvalues(evr)
    fig_scree.update_layout(title="Step 5 — Scree plot (shared PCA)")

    # ------------------------------------------------------------------ Step 6
    container_pc = container_pooled.with_features(
        shared_scores,
        names=[f"PC{i + 1}" for i in range(n_components)],
        new_dim_name="component",
    )
    container_traj = container_pc.unstack("obs")
    trajectories = container_traj.X[..., :3]
    trial_labels = np.asarray(container_traj.y).astype(int)

    # ------------------------------------------------------------------ Step 7: trajectories
    unique_conds = sorted(set(trial_labels.tolist()))
    mean_trajs, sem_trajs = [], []
    for cond in unique_conds:
        mask = trial_labels == cond
        cond_trajs = trajectories[mask]
        mean_trajs.append(cond_trajs.mean(axis=0))
        sem_trajs.append(cond_trajs.std(axis=0) / np.sqrt(max(cond_trajs.shape[0], 1)))
    X_centroid = np.stack(mean_trajs)
    sem_centroid = np.stack(sem_trajs)
    cond_names = np.array([LABEL_NAMES[c] for c in unique_conds])

    fig_traj_2d = plot_trajectory(
        X=X_centroid[..., :2],
        times=times,
        labels=cond_names,
        sem=sem_centroid[..., :2],
        color_map=CONDITION_COLORS,
        title="Step 7 — Group-mean trajectories (PC1–PC2, SEM envelope)",
        dimensions=2,
        smooth_window=8,
    )
    fig_traj_3d = plot_trajectory(
        X=X_centroid,
        times=times,
        labels=cond_names,
        color_map=CONDITION_COLORS,
        title="Step 7 — Group-mean trajectories (PC1–PC2–PC3)",
        dimensions=3,
        smooth_window=8,
    )

    # ------------------------------------------------------------------ Step 8: separation + stats
    print("Computing separation timecourses (centroid + Mahalanobis) …")
    sep_curves = compute_separation_timecourses(
        trajectories, trial_labels, methods=("centroid", "mahalanobis")
    )
    pair_scalars = compute_separation_pair_scalars(
        trajectories,
        times,
        subject_ids,
        trial_labels,
        methods=("centroid", "mahalanobis"),
    )

    fig_sep_c = plot_trajectory_separation(
        sep_curves["centroid"],
        times=times,
        title="Step 8 — Separation (centroid)",
    )
    fig_sep_m = plot_trajectory_separation(
        sep_curves["mahalanobis"],
        times=times,
        title="Step 8 — Separation (Mahalanobis)",
    )

    _peak_mask = (pair_scalars["method"] == "centroid") & (
        pair_scalars["metric"] == "peak_separation"
    )
    peak_by_pair = (
        pair_scalars.loc[_peak_mask].groupby("pair")["value"].agg(["mean", "std"])
    )
    fig_peak = plot_bar(
        scores=peak_by_pair["mean"],
        errors=peak_by_pair["std"],
        title="Step 8 — Peak separation per pair (centroid)",
        yaxis_title="peak distance (a.u.)",
    )

    _auc_mask = pair_scalars["metric"] == "auc_separation"
    auc_pivot = pair_scalars.loc[_auc_mask].pivot_table(
        index="pair", columns="method", values="value", aggfunc="mean"
    )
    fig_auc_heat = plot_heatmap(
        auc_pivot,
        annotate=True,
        annotation_format=".2f",
        title="Step 8 — Mean AUC per pair × method",
        xaxis_title="method",
        yaxis_title="condition pair",
    )

    # Paired t-tests with FDR over AUC pairs (paper §7.5)
    _mask = (pair_scalars["method"] == "centroid") & (
        pair_scalars["metric"] == "auc_separation"
    )
    auc_long = pair_scalars.loc[_mask].rename(columns={"pair": "condition"})[
        ["subject", "condition", "metric", "value"]
    ]
    paired_stats = paired_condition_stats(
        auc_long, conditions=sorted(auc_long["condition"].unique())
    )

    # ------------------------------------------------------------------ Step 9: metrics
    print("Computing all 13 per-trial / per-condition scalars …")
    scalar_metrics = compute_per_trial_scalars(
        trajectories, times, subject_ids, trial_labels
    )
    condition_scalars = compute_per_condition_scalars(
        trajectories, subject_ids, trial_labels
    )
    scalar_metrics_all = pd.concat(
        [
            scalar_metrics.assign(level="trial"),
            condition_scalars.assign(level="condition"),
        ],
        ignore_index=True,
    )
    scalar_metrics_all["condition_name"] = scalar_metrics_all["condition"].map(
        LABEL_NAMES
    )

    speed_mean, speed_sem = [], []
    for cond in unique_conds:
        mask = trial_labels == cond
        cond_speed = trajectory_speed(trajectories[mask], time=times)
        speed_mean.append(np.nanmean(cond_speed, axis=0))
        speed_sem.append(np.nanstd(cond_speed, axis=0) / np.sqrt(mask.sum()))

    speed_dict = {LABEL_NAMES[c]: m for c, m in zip(unique_conds, speed_mean)}
    fig_speed = plot_trajectory_metric_series(
        speed_dict,
        times=times[1:],
        title="Step 9 — Trajectory speed per condition (Table 2 row 1)",
        ylabel="speed (a.u./s)",
    )

    path_dict = {
        LABEL_NAMES[c]: float(
            np.nanmean(trajectory_path_length(trajectories[trial_labels == c]))
        )
        for c in unique_conds
    }
    fig_path = plot_bar(
        pd.Series(path_dict),
        title="Step 9 — Mean path length per condition",
        yaxis_title="path length (a.u.)",
    )

    spread_dict = {
        LABEL_NAMES[c]: trajectory_dispersion(trajectories[trial_labels == c])
        for c in unique_conds
    }
    fig_spread = plot_trajectory_metric_series(
        spread_dict,
        times=times,
        title="Step 9 — Within-condition spread (dispersion)",
        ylabel="within-group spread",
    )

    # Comet plot: Left Hand trajectory colour-coded by speed
    left_idx = unique_conds.index(3)
    fig_comet = plot_trajectory(
        X=X_centroid[left_idx, :, :3],
        times=times,
        values=speed_dict["Left Hand (Exec)"],
        title="Step 9 — Dynamical State: Trajectory Color-Coded by Neural Speed",
        dimensions=3,
        show_markers=False,
        smooth_window=5,
    )
    fig_comet.update_traces(line=dict(colorscale="Inferno", width=6))

    # Phase portrait: PC1 position vs velocity
    pc1_position = X_centroid[..., 0]
    pc1_velocity = np.gradient(pc1_position, axis=-1)
    phase_space = np.stack([pc1_position, pc1_velocity], axis=-1)
    fig_phase = plot_trajectory(
        X=phase_space,
        times=times,
        labels=cond_names,
        color_map=CONDITION_COLORS,
        title="Step 9 — Dynamical Phase Portrait (PC1 Position vs Velocity)",
        dimensions=2,
        show_markers=False,
        smooth_window=5,
    )
    fig_phase.update_xaxes(title_text="PC1 Position (Amplitude)")
    fig_phase.update_yaxes(title_text="PC1 Velocity (Rate of Change)")

    # Tortuosity violin distribution
    fig_tort = plot_distribution_groups(
        data=scalar_metrics_all[scalar_metrics_all["metric"] == "tortuosity"],
        value_col="value",
        group_col="condition_name",
        title='Step 9 — Trajectory Tortuosity (Neural "Wandering")',
    )
    fig_tort.update_yaxes(title_text="Ratio (Path Length / Displacement)")

    # ------------------------------------------------------------------ Step 9.5: advanced kinematics
    focus_cond = next(
        (
            c
            for c, name in LABEL_NAMES.items()
            if "exec" in name.lower() and "feet" in name.lower()
        ),
        unique_conds[-1],
    )
    focus_name = LABEL_NAMES[focus_cond]
    focus_mask = trial_labels == focus_cond
    focus_trajs = trajectories[focus_mask]  # already sliced to [:, :, :3]
    print(f"Step 9.5 — analysing '{focus_name}' ({focus_mask.sum()} trials)")

    dt = float(np.diff(times).mean())

    speed_trials_95 = trajectory_speed(focus_trajs, time=times)
    speed_curve_95 = np.nanmean(speed_trials_95, axis=0)
    t_speed_95 = times[:-1]

    jerk_trials_95 = trajectory_jerk(focus_trajs, dt=dt)
    jerk_curve_95 = np.nanmean(jerk_trials_95, axis=0)

    angle_trials_95 = trajectory_turning_angle(focus_trajs)
    angle_curve_95 = np.nanmean(np.degrees(angle_trials_95), axis=0)
    t_angle_95 = times[1:-1]

    disp_curve_95 = trajectory_dispersion(focus_trajs)

    speed_min_idx, _ = find_peaks(-speed_curve_95, distance=5)
    jerk_max_idx, _ = find_peaks(
        jerk_curve_95, distance=5, height=np.percentile(jerk_curve_95, 70)
    )
    disp_deriv_95 = np.gradient(disp_curve_95, dt)
    disp_max_idx, _ = find_peaks(
        disp_deriv_95, distance=5, height=np.percentile(disp_deriv_95, 70)
    )

    _95_metrics = {
        "Speed": (speed_curve_95, t_speed_95),
        "Jerk": (jerk_curve_95, times),
        "Turning Angle (°)": (angle_curve_95, t_angle_95),
        "Intra-Condition Dispersion": (disp_curve_95, times),
    }
    _95_ylabels = {
        "Speed": "speed (a.u./s)",
        "Jerk": "jerk (a.u./s³)",
        "Turning Angle (°)": "angle (°)",
        "Intra-Condition Dispersion": "spread (a.u.)",
    }

    kin_figs: dict = {}
    for metric_label, (curve, t_ax) in _95_metrics.items():
        n = min(len(t_ax), len(curve))
        fig = plot_trajectory_metric_series(
            {focus_name: curve[:n]},
            times=t_ax[:n],
            title=f"Step 9.5 — {metric_label} ({focus_name})",
            ylabel=_95_ylabels[metric_label],
        )
        if metric_label == "Speed":
            for idx in speed_min_idx:
                if idx < len(t_speed_95):
                    fig.add_vline(
                        x=float(t_speed_95[idx]),
                        line_dash="dash",
                        line_color="red",
                        line_width=1.5,
                        annotation_text="slow",
                        annotation_position="top right",
                    )
        elif metric_label == "Jerk":
            for idx in jerk_max_idx:
                if idx < len(times):
                    fig.add_vline(
                        x=float(times[idx]),
                        line_dash="dash",
                        line_color="#1f77b4",
                        line_width=1.5,
                        annotation_text="snap",
                        annotation_position="top right",
                    )
        elif metric_label == "Intra-Condition Dispersion":
            for idx in disp_max_idx:
                if idx < len(times):
                    fig.add_vline(
                        x=float(times[idx]),
                        line_dash="dash",
                        line_color="#9467bd",
                        line_width=1.5,
                        annotation_text="diverge",
                        annotation_position="top right",
                    )
        kin_figs[metric_label] = fig

    # ------------------------------------------------------------------ Step 10: validate
    print(f"Permutation null on separation AUC (n_perm={n_perm}) …")
    score_n = min(2000, shared_scores.shape[0])
    score_idx = rng.choice(shared_scores.shape[0], size=score_n, replace=False)
    score_payload = shared_reducer.score(
        X_emb=shared_scores[score_idx, :3],
        X=container_pooled.X[score_idx],
        n_neighbors=min(20, score_n - 2),
        metrics=["trustworthiness", "continuity"],
    )
    quality = score_payload.get("metrics", {})
    obs_auc, null_auc = permutation_null_separation_auc(
        trajectories,
        times,
        trial_labels,
        group_a=[3, 4],
        group_b=[7, 8],
        n_perm=n_perm,
        rng=rng,
        window=analysis_window,
    )
    p_value = float((np.sum(null_auc >= obs_auc) + 1) / (len(null_auc) + 1))
    print(
        f"trustworthiness={quality.get('trustworthiness', float('nan')):.3f}, "
        f"continuity={quality.get('continuity', float('nan')):.3f}, "
        f"p(AUC)={p_value:.4f}"
    )

    fig_shepard = plot_shepard_diagram(
        X_orig=container_pooled.X[score_idx],
        X_emb=shared_scores[score_idx, :3],
        sample_size=300,
        title="Step 10 — Shepard diagram (3D vs full sensor space)",
    )

    fig_null = plot_null_interval_summary(
        observed=obs_auc,
        null_distribution=null_auc,
        title=(
            f"Step 10 — Permutation null on AUC (Unilateral vs Bilateral)<br>"
            f"Empirical p-value: {p_value:.3f}"
        ),
        xaxis_title="AUC under label shuffle",
    )

    # ------------------------------------------------------------------ Composite
    print("Assembling 5-panel paper composite …")
    fig_paper = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "scene"}],
            [{"type": "xy", "colspan": 2}, None, {"type": "xy"}],
        ],
        subplot_titles=(
            "A — Scree",
            "B — Trial-mean PC1–PC2",
            "C — 3D trajectories",
            "D — Speed timecourse",
            "E — Separation (centroid + Mahalanobis)",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    # Panel A: scree
    fig_paper.add_trace(
        go.Bar(
            x=[f"PC{i + 1}" for i in range(len(evr))],
            y=evr,
            name="Var.",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Panel B: trial-mean PC1–PC2 scatter
    trial_means = trajectories.mean(axis=1)
    for cond in unique_conds:
        mask = trial_labels == cond
        fig_paper.add_trace(
            go.Scatter(
                x=trial_means[mask, 0],
                y=trial_means[mask, 1],
                mode="markers",
                marker=dict(
                    color=CONDITION_COLORS.get(cond, "#444"), size=4, opacity=0.5
                ),
                name=LABEL_NAMES[cond],
                legendgroup=LABEL_NAMES[cond],
            ),
            row=1,
            col=2,
        )

    # Panel C: 3D group-mean trajectories
    for cond, traj in zip(unique_conds, mean_trajs):
        fig_paper.add_trace(
            go.Scatter3d(
                x=traj[:, 0],
                y=traj[:, 1],
                z=traj[:, 2],
                mode="lines",
                line=dict(color=CONDITION_COLORS.get(cond, "#444"), width=5),
                name=LABEL_NAMES[cond],
                legendgroup=LABEL_NAMES[cond],
                showlegend=False,
            ),
            row=1,
            col=3,
        )

    # Panel D: speed timecourse with SEM
    for cond, m, s in zip(unique_conds, speed_mean, speed_sem):
        fig_paper.add_trace(
            go.Scatter(
                x=times[1:],
                y=m,
                mode="lines",
                line=dict(color=CONDITION_COLORS.get(cond, "#444"), width=2),
                error_y=dict(type="data", array=s, visible=True, thickness=0.5),
                name=LABEL_NAMES[cond],
                legendgroup=LABEL_NAMES[cond],
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Panel E: separation timecourses (centroid + Mahalanobis overlaid)
    for method, dash in [("centroid", "solid"), ("mahalanobis", "dash")]:
        for (a, b), curve in sep_curves[method].items():
            fig_paper.add_trace(
                go.Scatter(
                    x=times,
                    y=curve,
                    mode="lines",
                    line=dict(width=2, dash=dash),
                    name=f"{LABEL_NAMES[a]} vs {LABEL_NAMES[b]} ({method})",
                ),
                row=2,
                col=3,
            )

    fig_paper.update_layout(
        title=f"PCA Neural Trajectory Analysis — EEGBCI ({len(subject_list)} subjects)",
        height=820,
        width=1400,
        template="plotly_white",
    )

    fig_paper.write_image(str(out / "paper_figure.png"), scale=2)
    fig_paper.write_image(str(out / "paper_figure.svg"))
    fig_paper.write_html(str(out / "paper_figure.html"), include_plotlyjs="inline")

    # ------------------------------------------------------------------ Report (10-step)
    report = Report(
        title=f"EEGBCI — PCA Neural Trajectories ({len(subject_list)} subjects)",
        asset_urls="inline",
    )

    # ── Overview ────────────────────────────────────────────────────────────
    sec_overview = Section("Overview", icon="O")
    sec_overview.add_element(
        MarkdownElement(
            f"This companion report runs the full **10-step PCA neural-trajectory workflow** "
            f"from the paper *'A Primer on Low-Dimensional Neural Dynamics'* at paper scale "
            f"({len(subject_list)} subjects).\n\n"
            f"| Parameter | Value |\n"
            f"|-----------|-------|\n"
            f"| Subjects | {len(subject_list)} |\n"
            f"| Conditions | {', '.join(LABEL_NAMES[c] for c in unique_conds)} |\n"
            f"| Components fit | {n_components} (3 used for visualization) |\n"
            f"| Cumulative variance @ 3 PCs | {float(np.sum(evr[:3])):.2%} |\n"
            f"| Participation ratio (PR) | {pr:.2f} |\n"
            f"| Trustworthiness | {quality.get('trustworthiness', float('nan')):.3f} |\n"
            f"| Continuity | {quality.get('continuity', float('nan')):.3f} |\n"
            f"| Permutation p (AUC, n_perm={n_perm}) | {p_value:.4f} |\n\n"
            f"> Each section below corresponds to one numbered step in the tutorial notebook "
            f"`tutorials/tutorial_pca_trajectories_eegbci.ipynb`."
        )
    )
    report.add_section(sec_overview)

    # ── Step 1 — Choose Your Representation ─────────────────────────────────
    sec1 = Section("Step 1 — Choose Your Representation", icon="1")
    sec1.add_element(
        MarkdownElement(
            "The first question in any neural trajectory analysis is: *what is the "
            '"state" of the brain at time t?*\n\n'
            "We choose the **sensor-space voltage pattern** — a vector of 64 simultaneous "
            "EEG channel amplitudes:\n\n"
            "**x**(t) = [v₁(t), v₂(t), …, v₆₄(t)]ᵀ ∈ ℝ⁶⁴\n\n"
            "Each time point is therefore a single point in a 64-dimensional state space, "
            "and each trial traces out a **trajectory** through that space. Dimensionality "
            "reduction (PCA) will reveal a low-dimensional manifold — typically 2–4 "
            "dimensions — that captures the dominant variance across trials and conditions.\n\n"
            "**Dataset:** PhysioNet EEGBCI — "
            f"{len(subject_list)} subjects, 4 motor-execution conditions:\n\n"
            "| Label | Condition | Laterality |\n"
            "|-------|-----------|------------|\n"
            "| 3 | Left Hand (Exec) | Unilateral |\n"
            "| 4 | Right Hand (Exec) | Unilateral |\n"
            "| 7 | Hands (Exec) | Bilateral |\n"
            "| 8 | Feet (Exec) | Bilateral |\n\n"
            "The 4 conditions form a 2×2 factorial design (Hand/Foot × Unilateral/Bilateral) "
            "that spans multiple axes of motor cortex organization, making it an ideal "
            "testbed for trajectory-based analysis."
        )
    )
    report.add_section(sec1)

    # ── Step 2 — Preprocess ─────────────────────────────────────────────────
    sec2 = Section("Step 2 — Preprocess & Normalize", icon="2")
    sec2.add_element(
        MarkdownElement(
            "> **Why preprocessing is critical for PCA:** PCA does not know what is "
            '"neural signal" and what is "noise" — it simply finds the directions of '
            "maximum variance. If you don't remove eye blinks and heartbeats, those huge "
            "artifacts will completely dominate Principal Component 1!\n\n"
            "Before extracting neural trajectories, raw EEG data must be cleaned and "
            "epoched. Three sequential phases are applied:\n\n"
            "**Phase 1 — Download & Preprocess (MNE-Python)**\n\n"
            "The `setup_data_bids` helper executes a state-of-the-art MNE preprocessing "
            "pipeline and writes the epoched data to a standardized BIDS directory:\n\n"
            "1. **Bandpass filter** 0.5–40 Hz (FIR) — attenuates slow drifts and "
            "broadband muscle noise. *(The 40 Hz lowpass naturally suppresses most "
            "broadband muscle artifacts.)*\n"
            "2. **Notch filter** 50/60 Hz — removes residual line noise.\n"
            "3. **ICA artifact removal** — 20-component ICA with `mne-icalabel` to "
            "automatically identify and subtract **blinks** and **heartbeats**.\n\n"
            "**Phase 2 — Load into coco-pipe**\n\n"
            "The preprocessed BIDS directory is parsed by `BIDSDataset`, which produces "
            "a `DataContainer` with shape `[Trials × Channels × Time]`.\n\n"
            "**Phase 3 — Crop, Baseline, and Z-score**\n\n"
            "- **Crop** to the analysis window and apply per-channel baseline correction "
            "to remove raw sensor drift.\n"
            "- **Z-score** each channel across observations to prevent high-amplitude "
            "sensors from dominating PCA variance.\n"
            "- **Subset** the strict task window (`0.0s–1.5s`) for PCA fitting, while "
            "keeping the wider plotting window for visualization."
        )
    )
    report.add_section(sec2)

    # ── Step 3 — Reshape ────────────────────────────────────────────────────
    sec3 = Section("Step 3 — Reshape into (Samples × Features)", icon="3")
    sec3.add_element(
        MarkdownElement(
            "To apply PCA, our 3-dimensional neural data must be reshaped into a flat "
            "2-dimensional matrix. Two crucial methodological choices govern this step:\n\n"
            "- **Single-Trial Preservation:** We do *not* average trials before PCA. "
            "Every single timepoint of every trial is treated as a separate independent "
            "observation, capturing true biological variance across individual repetitions.\n"
            "- **Subject Normalization:** We explore two model types — a *Shared PCA* "
            "(all subjects pooled) and *Per-Subject PCAs* — to disentangle between-subject "
            "differences from within-condition neural dynamics.\n\n"
            "> **The Tensor Flattening Math:** When loaded, data is a 3D tensor "
            "`[Total Trials (T), Channels (C), Time (N)]`, where T = N_trials × N_subjects. "
            "`DataContainer.stack()` flattens this to `[(T × N) Observations, C Features]`. "
            "Because `coco-pipe` tracks the metadata expansion, we can cleanly `unstack()` "
            "PCA scores back into 3D trajectories later — losslessly.\n\n"
            "The data is stacked **twice**: once on the pooled container (for the Shared "
            "PCA in Step 4a), and once per subject (for Per-Subject PCAs in Step 4b)."
        )
    )
    report.add_section(sec3)

    # ── Step 4 — Fit PCA ─────────────────────────────────────────────────────
    sec4 = Section("Step 4 — Fit PCA Models", icon="4")
    sec4.add_element(
        MarkdownElement(
            "Because we are dealing with multiple subjects, we must be careful about "
            "group-level alignment. We solve this by fitting **two different types of "
            "PCA models**:\n\n"
            "#### 4a. Shared PCA (For Group-Level Figures)\n\n"
            "One global PCA fit on the **pooled** subject data. Every group-mean trajectory "
            "figure lives in this shared basis — spatial topographies mean the same thing "
            "across all subjects.\n\n"
            "#### 4b. Per-Subject PCA (For Scalar Statistics)\n\n"
            "A separate PCA for each individual subject. These *per-subject* scores are "
            "used exclusively to compute scalar kinematics (speed, path length, etc.) in "
            "Step 9, ensuring statistics capture true within-subject variance.\n\n"
            '> **The Group-Level Alignment Caveat:** Subject 1\'s "PC1" is not necessarily '
            'the same spatial direction as Subject 2\'s "PC1" — components can swap order '
            "or rotate based on individual head geometries. `flip_pc_scores_for_consistency` "
            "fixes *sign* flips, but cannot fix component swaps. This is why we maintain "
            "both a Shared PCA (for plotting) and Per-Subject PCAs (for statistics)."
        )
    )
    report.add_section(sec4)

    # ── Step 5 — Select components ──────────────────────────────────────────
    sec5 = Section("Step 5 — Select Components", icon="5")
    sec5.add_element(
        MarkdownElement(
            "How many Principal Components do we actually need? For trajectory "
            "visualization, **2–3 PCs typically suffice** for EEG/MEG. We evaluate "
            "this using two complementary methods:\n\n"
            "1. **Scree Plot** — visual plot of the Explained Variance Ratio (EVR) per "
            'component. Look for the "elbow" where additional PCs yield diminishing '
            "returns.\n"
            "2. **Participation Ratio (PR)** — a robust mathematical metric giving a "
            "single number representing the *effective dimensionality* of the dataset. "
            "Unlike an arbitrary variance cutoff, PR mathematically quantifies how many "
            "dimensions the neural data is *actually* using.\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Cumulative variance @ 3 PCs | **{float(np.sum(evr[:3])):.2%}** |\n"
            f"| Cumulative variance @ 5 PCs | **{float(np.sum(evr[:5])):.2%}** |\n"
            f"| Participation ratio (PR) | **{pr:.2f}** |\n\n"
            f"A PR of {pr:.2f} means the 64-channel neural trajectories effectively live "
            f"in a ~{pr:.1f}-dimensional space. We retain **3 PCs** for visualization and "
            f"10 PCs for downstream scalar metrics."
        )
    )
    sec5.add_element(PlotlyElement(fig_scree, height="420px"))
    report.add_section(sec5)

    # ── Step 6 — Project, unstack, baseline ─────────────────────────────────
    sec6 = Section("Step 6 — Project, Unstack & Baseline", icon="6")
    sec6.add_element(
        MarkdownElement(
            "> **Why do we need post-processing in PC-space?** PCA is a purely mathematical "
            "rotation. It does not guarantee that trajectories start at the origin (0, 0, 0), "
            "nor does it guarantee consistent eigenvector sign across subjects. We enforce "
            "these constraints manually.\n\n"
            "Now that our PCA basis is defined, we project the data back into biological "
            "structure `(trials × time × components)` via `with_features` + `unstack()`. "
            "Two post-processing corrections follow:\n\n"
            "**1. PC-Space Baselining** (`apply_pca_score_baseline`)\n\n"
            "Subtracts each trial's mean over the true pre-stimulus window, anchoring the "
            "start of each trajectory at **(0, 0, 0)** exactly at stimulus onset. "
            "*(Note: Step 2 centered raw channels to correct sensor drift; this secondary "
            "pass anchors the components themselves after the PCA rotation.)*\n\n"
            "**2. Sign-Alignment — Per-Subject Only** (`flip_pc_scores_for_consistency`)\n\n"
            'PCA eigenvectors are arbitrarily signed (a PC pointing "up" is '
            'mathematically identical to one pointing "down"). We flip the signs of the '
            "**per-subject** scores so that scalar statistics in Step 9 are stable and "
            "directly comparable across subjects. The shared PCA scores are not sign-flipped "
            "because a single, globally shared basis has no cross-subject sign inconsistencies."
        )
    )
    report.add_section(sec6)

    # ── Step 7 — Group-mean trajectories ────────────────────────────────────
    sec7 = Section("Step 7 — Group-Mean Trajectories", icon="7")
    sec7.add_element(
        MarkdownElement(
            "This is the headline state-space visualization of the pipeline. We compute "
            "two quantities for each condition:\n\n"
            "- **Centroid Trajectory:** Mathematical average of all single trials belonging "
            "to a condition (analogous to an ERP, but calculated in component space).\n"
            "- **Uncertainty Envelope:** Standard Error of the Mean (SEM) across trials.\n\n"
            "> **How to interpret these interactive plots:**\n"
            "> - **2D (PC1 vs PC2):** The translucent shading represents the SEM envelope — "
            "how stable the neural representation is at each moment.\n"
            "> - **3D (PC1 vs PC2 vs PC3):** Click and drag to rotate. Examine from "
            "different angles to spot when and where condition representations diverge.\n"
            "> - **The Baseline Anchor:** All trajectories originate tightly near (0, 0, 0) "
            "before branching out in response to the task — a direct consequence of the "
            "pre-stimulus PC-space baseline from Step 6."
        )
    )
    sec7.add_element(PlotlyElement(fig_traj_2d, height="520px"))
    sec7.add_element(PlotlyElement(fig_traj_3d, height="620px"))
    report.add_section(sec7)

    # ── Step 8 — Compare conditions ─────────────────────────────────────────
    sec8 = Section("Step 8 — Compare Conditions", icon="8")
    sec8.add_element(
        MarkdownElement(
            "To quantify how distinct different cognitive tasks are, we calculate the "
            "geometric **trajectory separation** over time under two distance definitions:\n\n"
            "- **Euclidean (Centroid):** Raw geometric distance between condition means "
            "in the PCA space.\n"
            "- **Mahalanobis:** Distance scaled by the trial-to-trial covariance, "
            "penalizing axes with high within-condition noise.\n\n"
            "> **How to interpret separation metrics:**\n"
            "> - **Timecourse:** Look for the precise moment the curve lifts from zero — "
            "this is the latency at which the brain successfully discriminates between tasks.\n"
            "> - **AUC (Area Under Curve):** Captures *sustained* discriminability over "
            "the entire task window.\n"
            "> - **Peak Separation:** Captures the maximum instantaneous neural divergence.\n\n"
            "We extract Peak and AUC per subject and run **paired t-tests with FDR "
            "correction** (`paired_condition_stats`) to rigorously test whether some task "
            "pairs are statistically more separable than others."
        )
    )
    sec8.add_element(PlotlyElement(fig_sep_c, height="420px"))
    sec8.add_element(PlotlyElement(fig_sep_m, height="420px"))
    sec8.add_element(PlotlyElement(fig_peak, height="380px"))
    sec8.add_element(PlotlyElement(fig_auc_heat, height="420px"))
    sec8.add_element(
        InteractiveTableElement(
            paired_stats, title="Paired t-tests on AUC (centroid, FDR)"
        )
    )
    report.add_section(sec8)

    # ── Step 9 — Trajectory metrics ─────────────────────────────────────────
    sec9 = Section("Step 9 — Trajectory Metrics & Dynamical Systems", icon="9")
    sec9.add_element(
        MarkdownElement(
            "While 3D visualizations are excellent for intuition, statistical rigor requires "
            "quantifying the geometric properties of these trajectories. We map the geometry "
            "of the neural state space to concrete cognitive interpretations using **Narrative "
            "Metrics** and classical **Dynamical Systems** analysis.\n\n"
            "> **The Cognitive Interpretation Cheat Sheet:**\n"
            "> - **Speed:** Moment-to-moment rate of change. Peak speed reflects rapid "
            "transitions between population states; often correlates with reaction time "
            "and decision commitment.\n"
            "> - **Comet Plot:** The group-mean trajectory color-coded by instantaneous "
            "speed. Bright segments = fast neural transitions; dark segments = slow, "
            "stable states.\n"
            "> - **Phase Portrait:** Plotting PC1 position against its own velocity. "
            "Spirals or rings mathematically prove the existence of **oscillatory limit "
            "cycles** in the brain.\n"
            "> - **Path Length:** Total distance traveled in state space — a proxy for "
            "the amount of representational change a condition demands (cognitive effort).\n"
            "> - **Tortuosity:** Ratio of total path length to straight-line displacement. "
            'High tortuosity indicates a "wandering" neural representation, e.g., due '
            "to task ambiguity.\n"
            "> - **Dispersion:** Across-trial variability. Low dispersion = highly "
            "consistent neural processing.\n\n"
            "*All 13 Table-2 metrics (acceleration, jerk, curvature, cohesion, intra-spread …) "
            "are computed per subject and exported to the appendix table below.*"
        )
    )
    sec9.add_element(PlotlyElement(fig_speed, height="400px"))
    sec9.add_element(PlotlyElement(fig_comet, height="600px"))
    sec9.add_element(PlotlyElement(fig_phase, height="500px"))
    sec9.add_element(PlotlyElement(fig_path, height="380px"))
    sec9.add_element(PlotlyElement(fig_tort, height="450px"))
    sec9.add_element(PlotlyElement(fig_spread, height="400px"))
    appendix = AccordionElement(
        "Appendix: full 13-metric grid (per-subject scalars)", open=False
    )
    appendix.add_element(
        InteractiveTableElement(
            scalar_metrics_all,
            title="All scalar metrics (level / metric / subject / condition / value)",
            selector_columns=["level", "metric", "condition_name"],
        )
    )
    sec9.add_element(appendix)
    report.add_section(sec9)

    # ── Step 9.5 — Advanced Kinematics & State Transitions ──────────────────
    sec95 = Section("Step 9.5 — Advanced Kinematics & State Transitions", icon="9½")
    sec95.add_element(
        MarkdownElement(
            f"Beyond the five narrative metrics, the *shape* of a trajectory through time "
            f"reveals **when** the brain switches representational states. We focus on "
            f"**{focus_name}** and overlay transition markers as vertical dashed lines:\n\n"
            "| Marker | Color | Cognitive Meaning |\n"
            "|--------|-------|-------------------|\n"
            "| Speed minima | Red | **Attractor states** — trajectory slows and settles into a stable holding pattern |\n"
            "| Jerk spikes | Blue | **Neural snaps** — abrupt, discrete state switches corresponding to cognitive decisions |\n"
            "| Dispersion jumps | Purple | **Trial divergence** — the moment individual trials begin deviating from the group mean |\n\n"
            "Four complementary descriptors characterize these dynamics:\n\n"
            "- **Speed Minima & Maxima.** A peak in speed indicates a rapid transition "
            "between states. A local minimum indicates the brain has entered a stable "
            "*attractor state* or holding pattern.\n"
            "- **Acceleration / Jerk.** Jerk is the derivative of acceleration. A sudden "
            'spike means the trajectory "snapped" into a new movement, often correlating '
            "with discrete cognitive decisions.\n"
            "- **Turning Angle.** The discrete angle between successive displacement vectors. "
            "A spike to 90–180° means the trajectory reversed direction — a hard shift in "
            "representational state.\n"
            "- **Intra-Condition Dispersion.** A sudden increase marks the exact moment where "
            "consistent sensory processing ends and variable decision-making begins."
        )
    )
    for fig in kin_figs.values():
        sec95.add_element(PlotlyElement(fig, height="400px"))
    report.add_section(sec95)

    # ── Step 10 — Validate ──────────────────────────────────────────────────
    sec10 = Section("Step 10 — Validate", icon="10")
    sec10.add_element(
        MarkdownElement(
            "Before drawing cognitive conclusions from lower-dimensional neural trajectories, "
            "we must rigorously validate that the dimensionality reduction did not destroy the "
            "true geometry of the neural state space, and that condition separations are not "
            "merely statistical noise.\n\n"
            "> **Validation Metrics:**\n"
            "> - **Trustworthiness:** Measures if PCA artificially crushed distant "
            "high-dimensional states close together in 3D (false neighbors). "
            "Scores > 0.85 are considered excellent.\n"
            "> - **Continuity:** Measures if PCA artificially tore close high-dimensional "
            "neighbors far apart in 3D.\n"
            "> - **Shepard Diagram:** Scatter of original 64D distances vs embedded 3D "
            "distances. A tight diagonal indicates perfect distance preservation.\n"
            "> - **Permutation Null (Separation AUC):** Condition labels are shuffled "
            "thousands of times to build a null distribution of trajectory separation. "
            "If the observed AUC exceeds the 95th percentile of the null, the neural "
            "separation is statistically significant (p < 0.05).\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Trustworthiness | **{quality.get('trustworthiness', float('nan')):.3f}** |\n"
            f"| Continuity | **{quality.get('continuity', float('nan')):.3f}** |\n"
            f"| Observed AUC (Unilateral vs Bilateral) | **{float(obs_auc):.3f}** |\n"
            f"| Empirical p-value (n_perm={n_perm}) | **{p_value:.4f}** |\n\n"
            f"The permutation null tests the Unilateral vs Bilateral contrast "
            f"(groups 3+4 vs 7+8) — condition labels are shuffled, separation AUC "
            f"recomputed, and the observed statistic is compared to the resulting null "
            f"distribution."
        )
    )
    sec10.add_element(PlotlyElement(fig_shepard, height="420px"))
    sec10.add_element(PlotlyElement(fig_null, height="380px"))
    report.add_section(sec10)

    # ── Paper composite ─────────────────────────────────────────────────────
    sec_fig = Section("Paper Composite — 5-Panel Figure", icon="F")
    sec_fig.add_element(
        MarkdownElement(
            "The 5-panel manuscript composite: scree (A), trial-mean PC1–PC2 scatter (B), "
            "3D group-mean trajectories (C), speed timecourse ± SEM (D), and separation "
            "timecourses for all condition pairs under centroid and Mahalanobis distance (E)."
        )
    )
    sec_fig.add_element(PlotlyElement(fig_paper, height="820px"))
    report.add_section(sec_fig)

    report.save(str(out / "report.html"))

    # ------------------------------------------------------------------ Save artifacts
    save_artifacts(
        artifacts_dir,
        shared_reducer=shared_reducer,
        trajectory_container=container_traj,
        per_subject_reducers=per_subject_reducers,
        scalar_metrics=scalar_metrics_all,
        continuous_metrics=pd.DataFrame(),
        separation_pair_scalars=pair_scalars,
        extra_csvs={"paired_stats": paired_stats},
    )

    return {
        "p_value": p_value,
        "observed_auc": float(obs_auc),
        "quality": quality,
        "n_subjects": len(subject_list),
        "output": out,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="*",
        default=None,
        help="Subject IDs (default: 1..109).",
    )
    parser.add_argument(
        "--output",
        default="outputs/paper/eegbci_109",
        help="Output root (default: outputs/paper/eegbci_109).",
    )
    parser.add_argument(
        "--bids-root",
        default="PhysioNet_EEGBCI/BIDS",
        help="BIDS root for the EEGBCI conversion.",
    )
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    result = run_paper_analysis(
        subjects=args.subjects,
        output=args.output,
        bids_root=args.bids_root,
        n_perm=args.n_perm,
        n_components=args.n_components,
        random_state=args.random_state,
    )
    print("=== done ===")
    print(result)


if __name__ == "__main__":
    main()
