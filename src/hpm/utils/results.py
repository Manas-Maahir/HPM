"""Aggregate per-run ``results.csv`` files into a single Excel workbook.

The standalone trainer (``train_hpm.py``) writes one ``results.csv`` per run under
``runs/hpm/{name}_{mode}_seed{seed}/results.csv``.  When several seeds are trained
(``--seeds 42,43,44``) this module rolls them up into ``runs/hpm/{name}_report.xlsx``:

* **one sheet per run** — the full epoch-by-epoch table;
* **BestPerRun** — the best epoch (by ``val/top1``) of every (mode, seed);
* **ModelSummary** — mean ± SD **across seeds** for each metric, per model;
* **Comparison** — HPM-vs-baseline significance tests on ``val/top1`` and ``val/f1_macro``.

Reporting a distribution + a statistical test (not a single number) is a hard
requirement of the project charter — see ``CLAUDE.md`` ("n > 1, always").
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Metric columns produced by train_hpm.py's _validate(); summarised across seeds.
_METRIC_COLS = [
    "val/top1",
    "val/top5",
    "val/f1_macro",
    "val/f1_weighted",
    "val/precision_macro",
    "val/recall_macro",
    "val/balanced_acc",
]

# Metrics we run an explicit HPM-vs-baseline significance test on.
_TEST_COLS = ["val/top1", "val/f1_macro"]

_SELECTION_METRIC = "val/top1"   # "best epoch" is the one that maximises this


def _run_dir(runs_dir: Path, name: str, mode: str, seed: int) -> Path:
    return runs_dir / f"{name}_{mode}_seed{seed}"


def _sheet_name(mode: str, seed: int) -> str:
    # Excel caps sheet names at 31 chars; "{mode}_s{seed}" is always well under.
    return f"{mode}_s{seed}"[:31]


def _best_row(df: pd.DataFrame) -> pd.Series:
    """Row of the epoch that maximises the selection metric (falls back to last)."""
    metric = _SELECTION_METRIC if _SELECTION_METRIC in df.columns else None
    if metric is None or df[metric].isna().all():
        return df.iloc[-1]
    return df.loc[df[metric].idxmax()]


def _collect(
    runs_dir: Path, name: str, modes: Sequence[str], seeds: Sequence[int]
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load every run's CSV.

    Returns (per_run_frames, best_per_run) where per_run_frames maps sheet name →
    full epoch table, and best_per_run is one row per (mode, seed) at its best epoch.
    """
    per_run: dict[str, pd.DataFrame] = {}
    best_rows: list[dict] = []

    for mode in modes:
        for seed in seeds:
            csv_path = _run_dir(runs_dir, name, mode, seed) / "results.csv"
            if not csv_path.exists():
                print(f"  [report] skip (no results.csv): {csv_path}")
                continue
            df = pd.read_csv(csv_path)
            if df.empty:
                print(f"  [report] skip (empty): {csv_path}")
                continue
            per_run[_sheet_name(mode, seed)] = df

            best = _best_row(df)
            row = {"model": mode, "seed": seed, "best_epoch": int(best.get("epoch", -1))}
            for col in _METRIC_COLS:
                if col in best:
                    row[col] = float(best[col])
            best_rows.append(row)

    best_df = pd.DataFrame(best_rows)
    return per_run, best_df


def _model_summary(best_df: pd.DataFrame) -> pd.DataFrame:
    """mean ± SD across seeds, per model, for every metric present."""
    if best_df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for mode, grp in best_df.groupby("model", sort=False):
        row: dict[str, object] = {"model": mode, "n_seeds": len(grp)}
        for col in _METRIC_COLS:
            if col not in grp:
                continue
            vals = grp[col].to_numpy(dtype=float)
            mean = float(np.mean(vals))
            # Sample SD (ddof=1); undefined for a single seed → 0.0 for a clean cell.
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            short = col.replace("val/", "")
            row[f"{short}_mean"] = round(mean, 6)
            row[f"{short}_sd"] = round(sd, 6)
            row[f"{short}"] = f"{mean:.4f} ± {sd:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def _comparison(best_df: pd.DataFrame) -> pd.DataFrame:
    """HPM-vs-baseline significance tests on the key metrics.

    Welch's t-test (unequal variance) plus a Mann-Whitney U fallback for the
    non-parametric read.  Needs >= 2 seeds per model; otherwise the test is skipped
    with a note so the sheet still explains why.
    """
    rows: list[dict] = []
    if best_df.empty or "hpm" not in set(best_df["model"]):
        return pd.DataFrame(rows)

    baselines = [m for m in best_df["model"].unique() if m != "hpm"]
    hpm = best_df[best_df["model"] == "hpm"]

    for metric in _TEST_COLS:
        if metric not in best_df.columns:
            continue
        a = hpm[metric].to_numpy(dtype=float)
        for base in baselines:
            b = best_df[best_df["model"] == base][metric].to_numpy(dtype=float)
            row: dict[str, object] = {
                "metric": metric.replace("val/", ""),
                "contrast": f"hpm vs {base}",
                "baseline": base,
                "hpm_mean": round(float(np.mean(a)), 6) if len(a) else np.nan,
                "baseline_mean": round(float(np.mean(b)), 6) if len(b) else np.nan,
                "delta": round(float(np.mean(a) - np.mean(b)), 6) if len(a) and len(b) else np.nan,
            }
            if len(a) >= 2 and len(b) >= 2:
                t_stat, t_p = stats.ttest_ind(a, b, equal_var=False)
                try:
                    _, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
                except ValueError:
                    u_p = np.nan
                row["t_stat"] = round(float(t_stat), 4)
                row["t_pvalue"] = round(float(t_p), 5)
                row["mannwhitney_pvalue"] = round(float(u_p), 5) if not np.isnan(u_p) else np.nan
                row["note"] = ""
            else:
                row["t_stat"] = np.nan
                row["t_pvalue"] = np.nan
                row["mannwhitney_pvalue"] = np.nan
                row["note"] = "need >= 2 seeds per model for a test"
            rows.append(row)

    return pd.DataFrame(rows)


def build_report(
    name: str,
    modes: Sequence[str],
    seeds: Sequence[int],
    runs_dir: Path | str = Path("runs") / "hpm",
) -> Path | None:
    """Build ``runs/hpm/{name}_report.xlsx`` from the runs' ``results.csv`` files.

    Returns the workbook path, or ``None`` if no results were found.
    """
    runs_dir = Path(runs_dir)
    per_run, best_df = _collect(runs_dir, name, modes, seeds)

    if not per_run:
        print(f"  [report] no results.csv found for '{name}' — nothing to write.")
        return None

    out_path = runs_dir / f"{name}_report.xlsx"
    summary_df = _model_summary(best_df)
    comparison_df = _comparison(best_df)

    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        # Summaries first so they open on top.
        if not summary_df.empty:
            summary_df.to_excel(xl, sheet_name="ModelSummary", index=False)
        if not best_df.empty:
            best_df.to_excel(xl, sheet_name="BestPerRun", index=False)
        if not comparison_df.empty:
            comparison_df.to_excel(xl, sheet_name="Comparison", index=False)
        for sheet, df in per_run.items():
            df.to_excel(xl, sheet_name=sheet, index=False)

    print(f"  [report] wrote {out_path.resolve()}  ({len(per_run)} runs)")
    return out_path
