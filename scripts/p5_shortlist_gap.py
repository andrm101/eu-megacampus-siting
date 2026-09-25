"""
P5 — Shortlist & Gap Analysis.

Applies tiered shortlisting to P4 suitability indices, quantifies structural
gaps for shortlisted regions, adds confidence bands via uncertainty scores,
and profiles the shortlists by country, archetype, and co-specialisation.

Tiers (applied to per-type min-max rescaled scores from P4):
  Tier 1 — High-Index : score >= 0.70
  Tier 2 — Candidate  : 0.60 <= score < 0.70

Structural readiness (SR): proportion of a type's features where the region
is above the 25th percentile (for "+" features) or below the 75th percentile
(for "-" features) of the full EU-242 sample. SR = 1.0 indicates no gaps.

Confidence band half-width: uncertainty_score * 0.10  (symmetric, clamped to [0,1]).

Outputs:
  analysis/p5_shortlist_tiered.csv   — 242 rows; tier per type + SR per type
  analysis/p5_gap_matrix.csv         — shortlisted regions; per-feature gap flags
  analysis/p5_country_type_counts.csv
  analysis/p5_summary_report.json
  figures/p5_tier_composition.png    — stacked bar: Tier1/2 counts per type
  figures/p5_uncertainty_profile.png — suitability vs uncertainty scatter (8 panels)
  figures/p5_gap_summary.png         — top gap features per type among shortlisted
  figures/p5_country_heatmap.png     — country x type shortlist intensity
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

np.random.seed(42)
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logging_config import pipeline_step  # noqa: E402

SCORES_PATH = ROOT / "data" / "gold" / "suitability_scores.parquet"
GOLD_PATH   = ROOT / "data" / "gold" / "megacampus_gold.parquet"
ANALYSIS    = ROOT / "analysis"
FIGURES     = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

TIER1_THRESHOLD = 0.70
TIER2_THRESHOLD = 0.60
UNCERTAINTY_HALFWIDTH = 0.10   # per unit of uncertainty_score
GAP_PERCENTILE = 25            # feature value below this pct = structural gap

_BG     = "#0a0f1e"
_BLUE   = "#00a8ff"
_AMBER  = "#ff8c42"
_GREEN  = "#00cc7a"
_RED    = "#ff3355"
_PURPLE = "#a855f7"
_CYAN   = "#66d9e8"
_GOLD   = "#f5a623"
_PINK   = "#fbbf24"

TYPE_LABELS = {
    "T1": "AI / ML Hub",
    "T2": "Biotechnology / Life Sciences",
    "T3": "Semiconductors / Advanced Electronics",
    "T4": "Cleantech / Green Technology",
    "T5": "Hyperscale Data Centre Hub",
    "T6": "HALEU / Advanced Nuclear",
    "T7": "Deep-Tech Robotics Campus",
    "T8": "Quantum / Photonics Anchor",
}

TYPE_COLORS = {
    "T1": _BLUE, "T2": _GREEN, "T3": _AMBER, "T4": _CYAN,
    "T5": _GOLD, "T6": _RED,   "T7": _PURPLE, "T8": _PINK,
}

# Type feature specs — must mirror p4 exactly (features, weights, directions)
TYPE_SPECS: dict[str, list[tuple[str, int, str]]] = {
    "T1": [
        ("score_talent",              3, "+"), ("score_infra",               3, "+"),
        ("lq_nace_j62j63",            3, "+"), ("hrst_per_1000",             2, "+"),
        ("ultrafast_broadband_pct",   2, "+"), ("business_rd_pct_gdp",       2, "+"),
        ("epo_patents_per_mio_pop",   2, "+"), ("doing_business_score",      2, "+"),
        ("cit_rate_pct",              1, "-"), ("electricity_price_eur_kwh", 1, "-"),
    ],
    "T2": [
        ("score_talent",              3, "+"), ("lq_nace_c21_m72",           3, "+"),
        ("hrst_per_1000",             3, "+"), ("gerd_hes_pct_gdp",          3, "+"),
        ("epo_patents_per_mio_pop",   2, "+"), ("score_cluster",             2, "+"),
        ("doing_business_score",      2, "+"), ("cit_rate_pct",              1, "-"),
        ("electricity_price_eur_kwh", 1, "-"),
    ],
    "T3": [
        ("lq_nace_c26",               3, "+"), ("electricity_price_eur_kwh", 3, "-"),
        ("score_talent",              2, "+"), ("hrst_per_1000",             2, "+"),
        ("rail_freight_ktonnes",      2, "+"), ("port_throughput_ktonnes",   2, "+"),
        ("doing_business_score",      2, "+"), ("artificial_land_pct",       1, "+"),
        ("cit_rate_pct",              1, "-"),
    ],
    "T4": [
        ("renewable_energy_share_pct", 3, "+"), ("lq_nace_d35_clean",        3, "+"),
        ("gerd_gov_pct_gdp",           2, "+"), ("esif_absorption_rate",      2, "+"),
        ("is_transition_region",       2, "+"), ("doing_business_score",      1, "+"),
        ("score_talent",               1, "+"), ("electricity_price_eur_kwh", 1, "-"),
    ],
    "T5": [
        ("electricity_price_eur_kwh",  3, "-"), ("ntc_import_mw",             3, "+"),
        ("renewable_energy_share_pct", 3, "+"), ("ultrafast_broadband_pct",   3, "+"),
        ("water_exploitation_index",   2, "-"), ("artificial_land_pct",       2, "+"),
        ("doing_business_score",       2, "+"), ("score_infra",               1, "+"),
    ],
    "T6": [
        ("haleu_smr_eligible_int",    3, "+"), ("nuclear_capacity_mw_log1p", 3, "+"),
        ("gerd_def_pct_gdp",          2, "+"), ("ntc_import_mw",             2, "+"),
        ("lq_nace_c21_m72",           1, "+"), ("score_cost",                1, "+"),
    ],
    "T7": [
        ("lq_nace_c26",               3, "+"), ("score_talent",              3, "+"),
        ("hrst_per_1000",             2, "+"), ("gerd_hes_pct_gdp",          2, "+"),
        ("epo_patents_per_mio_pop",   2, "+"), ("score_cluster",             2, "+"),
        ("doing_business_score",      2, "+"), ("rail_freight_ktonnes",      1, "+"),
    ],
    "T8": [
        ("quantum_programme_tier",    3, "+"), ("eurohpc_proximity_score",   3, "+"),
        ("score_talent",              3, "+"), ("gerd_hes_pct_gdp",          2, "+"),
        ("epo_patents_per_mio_pop",   2, "+"), ("ultrafast_broadband_pct",   2, "+"),
        ("score_cluster",             1, "+"),
    ],
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _minmax(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0.5, index=s.index, dtype=float)
    return (s - mn) / (mx - mn)


def _get_feature(gold: pd.DataFrame, feat: str) -> pd.Series:
    if feat not in gold.columns:
        return pd.Series(np.nan, index=gold.index)
    col = gold[feat]
    if col.dtype == bool or col.dtype == object:
        col = col.astype(float)
    return pd.to_numeric(col, errors="coerce")


def _prepare_gold(gold: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns required for gap analysis (mirrors p4 derivations)."""
    g = gold.copy()
    if "nuclear_capacity_mw" in g.columns:
        g["nuclear_capacity_mw_log1p"] = np.log1p(g["nuclear_capacity_mw"].fillna(0))
    if "haleu_smr_eligible" in g.columns:
        g["haleu_smr_eligible_int"] = g["haleu_smr_eligible"].astype(float)
    if "is_transition_region" in g.columns:
        g["is_transition_region"] = g["is_transition_region"].astype(float)
    if "quantum_programme_tier" in g.columns:
        g["quantum_programme_tier"] = pd.to_numeric(
            g["quantum_programme_tier"], errors="coerce").fillna(0) / 3.0
    return g


# ─── Tiered shortlisting ─────────────────────────────────────────────────────

def compute_tiers(scores: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with tier labels (1, 2, or NaN) per type, plus CI bounds."""
    rows = {}
    for tid in TYPE_LABELS:
        col = f"suitability_{tid}"
        unc = f"uncertainty_{tid}"
        s = scores[col]
        u = scores[unc] if unc in scores.columns else pd.Series(0.0, index=scores.index)

        tier = pd.Series(np.nan, index=scores.index, name=f"tier_{tid}", dtype="object")
        tier.loc[s >= TIER1_THRESHOLD] = "T1_high_index"
        tier.loc[(s >= TIER2_THRESHOLD) & (s < TIER1_THRESHOLD)] = "T2_candidate"

        ci_half = u * UNCERTAINTY_HALFWIDTH
        lb = (s - ci_half).clip(lower=0.0)
        ub = (s + ci_half).clip(upper=1.0)

        rows[f"tier_{tid}"]  = tier
        rows[f"ci_lo_{tid}"] = lb.round(4)
        rows[f"ci_hi_{tid}"] = ub.round(4)

    return pd.DataFrame(rows, index=scores.index)


# ─── Structural readiness ─────────────────────────────────────────────────────

def compute_structural_readiness(
        gold: pd.DataFrame,
        scores: pd.DataFrame
) -> pd.DataFrame:
    """
    For each type, compute per-region structural readiness (SR) = proportion of
    type features where the region is NOT in gap territory.

    Gap territory (for "+" features): below GAP_PERCENTILE-th pct of full sample.
    Gap territory (for "-" features): above (100 - GAP_PERCENTILE)-th pct.
    """
    sr_cols: dict[str, pd.Series] = {}
    gap_rows: list[dict] = []

    for tid, specs in TYPE_SPECS.items():
        n_feats = len(specs)
        feat_above_threshold = pd.Series(0.0, index=gold.index)

        for feat, w, direction in specs:
            vals = _get_feature(gold, feat).fillna(_get_feature(gold, feat).median()
                                                   if _get_feature(gold, feat).notna().any()
                                                   else 0.0)
            if direction == "+":
                threshold_val = np.nanpercentile(vals, GAP_PERCENTILE)
                in_gap = vals < threshold_val
            else:
                threshold_val = np.nanpercentile(vals, 100 - GAP_PERCENTILE)
                in_gap = vals > threshold_val

            feat_above_threshold += (~in_gap).astype(float)

            # Record gap flags for shortlisted regions
            shortlisted = scores[f"suitability_{tid}"] >= TIER2_THRESHOLD
            for nuts2 in gold.index[shortlisted & in_gap]:
                gap_rows.append({
                    "nuts2_code": nuts2,
                    "type_id": tid,
                    "feature": feat,
                    "weight": w,
                    "direction": direction,
                    "threshold_val": round(float(threshold_val), 4),
                    "actual_val": round(float(vals.loc[nuts2]), 4),
                })

        sr_cols[f"sr_{tid}"] = (feat_above_threshold / n_feats).round(4)

    sr_df = pd.DataFrame(sr_cols, index=gold.index)
    gap_df = pd.DataFrame(gap_rows)
    return sr_df, gap_df


# ─── Figures ─────────────────────────────────────────────────────────────────

def fig_tier_composition(tiers_df: pd.DataFrame) -> None:
    """Stacked horizontal bar: Tier1 (high-index) + Tier2 (candidate) per type."""
    tier1_counts = {}
    tier2_counts = {}
    for tid in TYPE_LABELS:
        col = f"tier_{tid}"
        tier1_counts[tid] = (tiers_df[col] == "T1_high_index").sum()
        tier2_counts[tid] = (tiers_df[col] == "T2_candidate").sum()

    tids   = list(TYPE_LABELS.keys())
    labels = [f"{t}: {TYPE_LABELS[t][:30]}" for t in tids]
    t1     = [tier1_counts[t] for t in tids]
    t2     = [tier2_counts[t] for t in tids]
    y      = np.arange(len(tids))

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=_BG)
    ax.set_facecolor("#0d1529")
    b1 = ax.barh(y, t1, color=[TYPE_COLORS[t] for t in tids], alpha=0.9, label="Tier 1: High-Index (≥0.70)")
    b2 = ax.barh(y, t2, left=t1, color=[TYPE_COLORS[t] for t in tids],
                 alpha=0.4, hatch="///", label="Tier 2: Candidate (0.60–0.70)")

    for i, (v1, v2) in enumerate(zip(t1, t2)):
        if v1 > 0:
            ax.text(v1 / 2, i, str(v1), ha="center", va="center",
                    color=_BG, fontsize=8.5, fontfamily="monospace", fontweight="bold")
        if v2 > 0:
            ax.text(v1 + v2 / 2, i, str(v2), ha="center", va="center",
                    color="#c0d0e8", fontsize=8, fontfamily="monospace")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, color="#c0d0e8", fontsize=9, fontfamily="monospace")
    ax.set_xlabel("Number of EU NUTS2 Regions", color="#c0d0e8", fontsize=9)
    ax.set_title("DISRUPTOR ECOSYSTEM SHORTLIST — TIER COMPOSITION\n"
                 "EU NUTS2 Regions by Suitability Index Tier",
                 color=_BLUE, fontsize=11, fontfamily="monospace", pad=10)
    ax.tick_params(colors="#c0d0e8", labelsize=8)
    ax.xaxis.label.set_color("#c0d0e8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3050")
    leg = ax.legend(loc="lower right", fontsize=8, framealpha=0.3,
                    facecolor="#0d1529", edgecolor="#1e3050", labelcolor="#c0d0e8")

    ax.axvline(242 * 0.30, color="#1e3050", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.text(242 * 0.30 + 1, len(tids) - 0.5, "30% of EU NUTS2",
            color="#c0d0e8", fontsize=7, fontfamily="monospace", alpha=0.7)

    plt.tight_layout()
    out = FIGURES / "p5_tier_composition.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print(f"    Saved: {out.relative_to(ROOT)}")


def fig_uncertainty_profile(scores_df: pd.DataFrame) -> None:
    """8-panel scatter: suitability vs uncertainty; colour = tier."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor=_BG)
    fig.suptitle("SUITABILITY INDEX vs. UNCERTAINTY SCORE — EU NUTS2 (n=242)",
                 color=_BLUE, fontsize=11, fontfamily="monospace", fontweight="bold", y=1.01)

    for i, (tid, label) in enumerate(TYPE_LABELS.items()):
        ax = axes[i // 4][i % 4]
        ax.set_facecolor("#0d1529")
        col = f"suitability_{tid}"
        unc = f"uncertainty_{tid}"
        sc  = scores_df[col]
        un  = scores_df[unc] if unc in scores_df.columns else pd.Series(0.0, index=scores_df.index)

        tier1 = sc >= TIER1_THRESHOLD
        tier2 = (sc >= TIER2_THRESHOLD) & ~tier1
        other = sc < TIER2_THRESHOLD

        ax.scatter(sc[other], un[other], c="#1e3a5f", s=15, alpha=0.5, linewidths=0)
        ax.scatter(sc[tier2], un[tier2], c=TYPE_COLORS[tid], s=25, alpha=0.7, linewidths=0)
        ax.scatter(sc[tier1], un[tier1], c=TYPE_COLORS[tid], s=45, alpha=1.0,
                   edgecolors="white", linewidths=0.5)

        ax.axvline(TIER1_THRESHOLD, color=_RED,  linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axvline(TIER2_THRESHOLD, color=_AMBER, linestyle=":",  linewidth=0.8, alpha=0.5)
        ax.set_title(f"{tid}: {label[:28]}", color=TYPE_COLORS[tid],
                     fontsize=8, fontfamily="monospace")
        ax.set_xlabel("Suitability Index", color="#c0d0e8", fontsize=7)
        ax.set_ylabel("Uncertainty Score", color="#c0d0e8", fontsize=7)
        ax.tick_params(colors="#c0d0e8", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e3050")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        n1 = tier1.sum(); n2 = tier2.sum()
        ax.text(0.02, 0.97, f"T1:{n1}  T2:{n2}", transform=ax.transAxes,
                color="#c0d0e8", fontsize=7, fontfamily="monospace", va="top")

    plt.tight_layout()
    out = FIGURES / "p5_uncertainty_profile.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print(f"    Saved: {out.relative_to(ROOT)}")


def fig_gap_summary(gap_df: pd.DataFrame) -> None:
    """Per-type: top gap features by weighted gap frequency among shortlisted regions."""
    if gap_df.empty:
        print("    SKIP gap summary: empty gap matrix")
        return

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), facecolor=_BG)
    fig.suptitle("TOP STRUCTURAL GAPS — SHORTLISTED REGIONS BY TYPE\n"
                 f"(Feature in bottom {GAP_PERCENTILE}th percentile of EU-242 sample)",
                 color=_BLUE, fontsize=11, fontfamily="monospace", fontweight="bold", y=1.01)

    for i, (tid, label) in enumerate(TYPE_LABELS.items()):
        ax = axes[i // 4][i % 4]
        ax.set_facecolor("#0d1529")

        sub = gap_df[gap_df["type_id"] == tid]
        if sub.empty:
            ax.set_visible(False)
            continue

        # Weighted gap frequency: sum(weight) for each feature across shortlisted regions
        gap_freq = sub.groupby("feature")["weight"].sum().sort_values(ascending=True)
        top = gap_freq.tail(6)  # top 6 most-common gaps

        bars = ax.barh(range(len(top)), top.values,
                       color=TYPE_COLORS[tid], alpha=0.8)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index, color="#c0d0e8", fontsize=7.5, fontfamily="monospace")
        ax.set_xlabel("Weighted gap frequency", color="#c0d0e8", fontsize=7)
        ax.set_title(f"{tid}: {label[:30]}", color=TYPE_COLORS[tid],
                     fontsize=8.5, fontfamily="monospace")
        ax.tick_params(colors="#c0d0e8", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e3050")

        n_regions = sub["nuts2_code"].nunique()
        ax.text(0.98, 0.04, f"n regions: {n_regions}", transform=ax.transAxes,
                ha="right", color="#c0d0e8", fontsize=7, fontfamily="monospace")

    plt.tight_layout()
    out = FIGURES / "p5_gap_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print(f"    Saved: {out.relative_to(ROOT)}")


def fig_country_heatmap(scores_df: pd.DataFrame, tiers_df: pd.DataFrame) -> None:
    """Country x type heatmap: number of Tier 1+2 shortlisted regions."""
    if "country_code" not in scores_df.columns:
        print("    SKIP country heatmap: country_code column missing")
        return

    combined = scores_df[["country_code"]].join(tiers_df)
    rows = []
    for tid in TYPE_LABELS:
        col = f"tier_{tid}"
        shortlisted = combined[col].isin(["T1_high_index", "T2_candidate"])
        counts = combined[shortlisted]["country_code"].value_counts()
        for cc, n in counts.items():
            rows.append({"country": cc, "type": tid, "n": n})

    if not rows:
        print("    SKIP country heatmap: no shortlisted regions")
        return

    ct = pd.DataFrame(rows).pivot_table(
        index="country", columns="type", values="n", fill_value=0)
    ct = ct.reindex(columns=list(TYPE_LABELS.keys()), fill_value=0)
    ct = ct.sort_values(by=list(TYPE_LABELS.keys()), ascending=False)

    fig, ax = plt.subplots(figsize=(13, max(8, len(ct) * 0.38 + 2)), facecolor=_BG)
    ax.set_facecolor("#0d1529")

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "intensity", ["#0d1529", "#1e3a5f", _BLUE, _CYAN])
    im = ax.imshow(ct.values, cmap=cmap, aspect="auto",
                   vmin=0, vmax=max(ct.values.max(), 1))

    ax.set_xticks(range(len(ct.columns)))
    ax.set_xticklabels(ct.columns, color="#c0d0e8", fontsize=9, fontfamily="monospace")
    ax.set_yticks(range(len(ct.index)))
    ax.set_yticklabels(ct.index, color="#c0d0e8", fontsize=8.5)

    for r in range(ct.shape[0]):
        for c in range(ct.shape[1]):
            v = ct.values[r, c]
            if v > 0:
                ax.text(c, r, str(int(v)), ha="center", va="center",
                        color="white" if v > 2 else "#c0d0e8",
                        fontsize=8, fontfamily="monospace")

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(colors="#c0d0e8", labelsize=7)
    cbar.set_label("Shortlisted regions (Tier 1+2)", color="#c0d0e8", fontsize=8)

    ax.set_title("SHORTLISTED REGIONS BY COUNTRY AND DISRUPTOR TYPE",
                 color=_BLUE, fontsize=11, fontfamily="monospace", pad=12)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e3050")
    ax.tick_params(colors="#c0d0e8")

    plt.tight_layout()
    out = FIGURES / "p5_country_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print(f"    Saved: {out.relative_to(ROOT)}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    with pipeline_step("p5_load"):
        scores = pd.read_parquet(SCORES_PATH)
        gold   = _prepare_gold(pd.read_parquet(GOLD_PATH))

    with pipeline_step("p5_tier_assignment"):
        tiers_df = compute_tiers(scores)
        out_df   = scores.join(tiers_df)

    with pipeline_step("p5_structural_readiness"):
        sr_df, gap_df = compute_structural_readiness(gold, scores)
        out_df = out_df.join(sr_df)

    with pipeline_step("p5_artifacts"):
        # Tiered shortlist (all 242 regions)
        out_df.to_csv(ANALYSIS / "p5_shortlist_tiered.csv")

        # Gap matrix
        if not gap_df.empty:
            gap_df.to_csv(ANALYSIS / "p5_gap_matrix.csv", index=False)

        # Country × type counts
        if "country_code" in scores.columns:
            country_rows = []
            for tid in TYPE_LABELS:
                col = f"tier_{tid}"
                shortlisted = tiers_df[col].isin(["T1_high_index", "T2_candidate"])
                t1_flag = tiers_df[col] == "T1_high_index"
                cc_counts = scores.loc[shortlisted, "country_code"].value_counts()
                for cc, n in cc_counts.items():
                    n_t1 = int((scores.index.isin(tiers_df.index[t1_flag]) &
                                (scores["country_code"] == cc)).sum())
                    country_rows.append({"type_id": tid, "country_code": cc,
                                         "n_tier1": n_t1, "n_tier2": int(n) - n_t1,
                                         "n_total": int(n)})
            pd.DataFrame(country_rows).to_csv(
                ANALYSIS / "p5_country_type_counts.csv", index=False)

        # Summary report
        summary: dict = {
            "tier1_threshold": TIER1_THRESHOLD,
            "tier2_threshold": TIER2_THRESHOLD,
            "uncertainty_halfwidth": UNCERTAINTY_HALFWIDTH,
            "gap_percentile": GAP_PERCENTILE,
            "n_regions": 242,
            "types": {},
        }
        for tid in TYPE_LABELS:
            col = f"tier_{tid}"
            t1 = int((tiers_df[col] == "T1_high_index").sum())
            t2 = int((tiers_df[col] == "T2_candidate").sum())
            sc = scores[f"suitability_{tid}"]
            unc = scores[f"uncertainty_{tid}"] if f"uncertainty_{tid}" in scores.columns \
                else pd.Series(0.0, index=scores.index)
            sr_median = float(sr_df[f"sr_{tid}"].median().round(3))
            gap_feats = gap_df[gap_df["type_id"] == tid]["feature"].value_counts().to_dict() \
                if not gap_df.empty else {}
            summary["types"][tid] = {
                "label": TYPE_LABELS[tid],
                "n_tier1": t1,
                "n_tier2": t2,
                "n_shortlisted": t1 + t2,
                "score_mean": round(float(sc.mean()), 3),
                "score_std": round(float(sc.std()), 3),
                "uncertainty_mean": round(float(unc.mean()), 3),
                "structural_readiness_median": sr_median,
                "top_gap_features": list(gap_feats.keys())[:3],
            }
        with open(ANALYSIS / "p5_summary_report.json", "w") as f:
            json.dump(summary, f, indent=2)

    with pipeline_step("p5_figures"):
        fig_tier_composition(tiers_df)
        fig_uncertainty_profile(scores)
        fig_gap_summary(gap_df)
        fig_country_heatmap(scores, tiers_df)

    # ── Gate check ────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("P5 SHORTLIST & GAP ANALYSIS — GATE CHECK")
    print("=" * 70)
    print(f"  Tier 1 threshold : >= {TIER1_THRESHOLD}")
    print(f"  Tier 2 threshold : >= {TIER2_THRESHOLD} and < {TIER1_THRESHOLD}")
    print(f"  Gap percentile   : < {GAP_PERCENTILE}th pct of EU-242 sample")
    print()
    print(f"  {'Type':<6} {'Label':<40} {'Tier1':>6} {'Tier2':>6} {'SR_med':>7}")
    print(f"  {'-'*6} {'-'*40} {'-'*6} {'-'*6} {'-'*7}")
    for tid in TYPE_LABELS:
        col = f"tier_{tid}"
        t1 = (tiers_df[col] == "T1_high_index").sum()
        t2 = (tiers_df[col] == "T2_candidate").sum()
        sr = sr_df[f"sr_{tid}"].median()
        print(f"  {tid:<6} {TYPE_LABELS[tid][:38]:<40} {t1:>6} {t2:>6} {sr:>7.3f}")
    print()
    print(f"  Gap matrix rows  : {len(gap_df)}")
    print(f"  Gap matrix types : {gap_df['type_id'].nunique() if not gap_df.empty else 0}")
    print()

    all_have_tier1 = all(
        (tiers_df[f"tier_{tid}"] == "T1_high_index").sum() >= 1
        for tid in TYPE_LABELS
    )
    gate_pass = out_df.shape[0] == 242 and all_have_tier1

    print(f"GATE_P5={'PASS' if gate_pass else 'FAIL'}")
    if not gate_pass:
        if out_df.shape[0] != 242:
            print(f"  REASON: expected 242 rows, got {out_df.shape[0]}")
        if not all_have_tier1:
            missing = [t for t in TYPE_LABELS
                       if (tiers_df[f"tier_{t}"] == "T1_high_index").sum() == 0]
            print(f"  REASON: no Tier 1 regions for {missing}")
    print("=" * 70)

    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
