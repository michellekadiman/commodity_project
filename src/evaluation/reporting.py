"""Tables, plots and summary statistics for architecture selection."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .dm_test import benjamini_hochberg, dm_tables, pooled_dm


def mse_table(errors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-commodity OOS MSE; 'ALL' column pools every (day, commodity)."""
    rows = {}
    for name, err in errors.items():
        per_com = (err**2).mean()
        per_com["ALL"] = float((err.to_numpy() ** 2).mean())
        rows[name] = per_com
    return pd.DataFrame(rows).T


def summary_table(
    errors: dict[str, pd.DataFrame], baseline: str = "VanillaAE"
) -> pd.DataFrame:
    """One row per architecture with the statistics that matter for
    selection: pooled MSE (and ratio to baseline), how broadly the
    improvement holds across commodities, and panel-level significance.
    """
    mses = mse_table(errors)
    stats_df, pvals_df = dm_tables(errors, baseline)
    base_mse = mses.loc[baseline].drop("ALL")

    rows = []
    for name in errors:
        per_com = mses.loc[name].drop("ALL")
        ratios = per_com / base_mse
        row = {
            "architecture": name,
            "pooled_mse": mses.loc[name, "ALL"],
            "pooled_mse_ratio": mses.loc[name, "ALL"] / mses.loc[baseline, "ALL"],
            "median_commodity_mse_ratio": float(ratios.median()),
            "n_commodities_better": int((ratios < 1).sum()),
        }
        if name == baseline:
            row.update(
                n_sig_better_fdr=np.nan,
                n_sig_worse_fdr=np.nan,
                median_dm_stat=np.nan,
                pooled_dm_stat=np.nan,
                pooled_dm_pvalue=np.nan,
            )
        else:
            pv = pvals_df.loc[name].to_numpy()
            st = stats_df.loc[name].to_numpy()
            sig = benjamini_hochberg(pv)
            pdm, pdm_p = pooled_dm(
                errors[name].reindex(errors[baseline].index), errors[baseline]
            )
            row.update(
                n_sig_better_fdr=int((sig & (st < 0)).sum()),
                n_sig_worse_fdr=int((sig & (st > 0)).sum()),
                median_dm_stat=float(np.nanmedian(st)),
                pooled_dm_stat=pdm,
                pooled_dm_pvalue=pdm_p,
            )
        rows.append(row)
    out = pd.DataFrame(rows).set_index("architecture")
    return out.sort_values("pooled_mse_ratio")


# ---- plots -----------------------------------------------------------------


def plot_dm_heatmap(stats_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 0.6 * len(stats_df) + 2.5))
    vmax = np.nanmax(np.abs(stats_df.to_numpy())) or 1.0
    sns.heatmap(
        stats_df.astype(float),
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".1f",
        annot_kws={"size": 7},
        cbar_kws={"label": "DM statistic"},
        ax=ax,
    )
    ax.set_title(
        "Diebold-Mariano statistics vs. VanillaAE\n"
        "(negative = architecture beats baseline; |DM| > 1.96 ~ 5% significance)"
    )
    ax.set_xlabel("Commodity")
    ax.set_ylabel("Architecture")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ranking(
    errors: dict[str, pd.DataFrame], summary: pd.DataFrame, baseline: str, path: Path
) -> None:
    """Architecture ranking: pooled MSE ratio plus the spread of
    per-commodity ratios, sorted best to worst."""
    mses = mse_table(errors)
    base = mses.loc[baseline].drop("ALL")
    order = summary.index.tolist()

    fig, ax = plt.subplots(figsize=(10, 6))
    for y, name in enumerate(order):
        ratios = (mses.loc[name].drop("ALL") / base).to_numpy()
        ax.scatter(ratios, np.full_like(ratios, y), alpha=0.45, s=22, color="steelblue")
        ax.scatter(
            summary.loc[name, "pooled_mse_ratio"],
            y,
            color="crimson",
            marker="D",
            s=70,
            zorder=3,
            label="pooled MSE ratio" if y == 0 else None,
        )
    ax.axvline(1.0, color="k", lw=1, ls="--")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("OOS MSE ratio vs. VanillaAE (< 1 is better)")
    ax.set_title(
        "Architecture ranking by out-of-sample forecast MSE\n"
        "(blue dots: individual commodities; red diamond: pooled)"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cumulative_ssd(
    errors: dict[str, pd.DataFrame], baseline: str, path: Path
) -> None:
    """Cumulative sum of (baseline squared error - architecture squared
    error), averaged across commodities. An upward-drifting line means the
    architecture consistently beats the baseline; useful for spotting
    improvements driven by a single episode (e.g. one volatility spike)
    rather than a stable edge -- important before committing to an
    architecture for the Shapley study.
    """
    base = errors[baseline]
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, err in errors.items():
        if name == baseline:
            continue
        d = (base**2 - err.reindex(base.index) ** 2).mean(axis=1).cumsum()
        ax.plot(d.index, d.to_numpy(), label=name, lw=1.2)
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_ylabel("Cumulative avg. squared-error advantage vs. VanillaAE")
    ax.set_title("Stability of forecast improvement over time (up = better)")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---- orchestration ----------------------------------------------------------


def write_all_outputs(
    errors: dict[str, pd.DataFrame],
    results_dir: str | Path,
    baseline: str = "VanillaAE",
) -> pd.DataFrame:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    mses = mse_table(errors)
    stats_df, pvals_df = dm_tables(errors, baseline)
    summary = summary_table(errors, baseline)

    mses.to_csv(results_dir / "mse_per_commodity.csv")
    stats_df.to_csv(results_dir / "dm_statistics.csv")
    pvals_df.to_csv(results_dir / "dm_pvalues.csv")
    summary.to_csv(results_dir / "summary.csv")

    plot_dm_heatmap(stats_df, results_dir / "dm_heatmap.png")
    plot_ranking(errors, summary, baseline, results_dir / "architecture_ranking.png")
    plot_cumulative_ssd(errors, baseline, results_dir / "cumulative_advantage.png")

    _write_summary_md(summary, results_dir / "summary.md", baseline)
    return summary


def _write_summary_md(summary: pd.DataFrame, path: Path, baseline: str) -> None:
    best = summary.index[0]
    lines = [
        "# Autoencoder architecture comparison — summary",
        "",
        "Out-of-sample one-step-ahead forecast accuracy of factor-augmented "
        f"AR(1) models, rolling 252-day windows, monthly refits. Baseline: {baseline}.",
        "",
        "Ranking (by pooled OOS MSE ratio vs. baseline; lower is better):",
        "",
        summary.round(4).to_markdown(),
        "",
        f"Best-ranked architecture: **{best}**.",
        "",
        "Selection guidance: prefer an architecture whose advantage is "
        "(1) significant in the pooled DM test, (2) broad across commodities "
        "(`n_commodities_better`, `n_sig_better_fdr`), and (3) stable over "
        "time (see `cumulative_advantage.png`). Per-commodity details are in "
        "`mse_per_commodity.csv`, `dm_statistics.csv`, `dm_pvalues.csv`.",
    ]
    path.write_text("\n".join(lines))
