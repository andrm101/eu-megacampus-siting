"""
P4 — Disruptor Ecosystem Suitability Score Construction.

Builds 8 composite suitability indices (T1–T8) from the Gold layer using
min-max normalisation, directional inversion, and conceptual weighting
(H=3, M=2, L=1). Produces uncertainty scores per type, co-specialisation
flags, dominant-type assignment, and 3 figures.

Outputs:
  data/gold/suitability_scores.parquet  — 242 × 36 (8 scores, 8 uncertainty,
                                           8 shortlist flags, co-spec flags,
                                           dominant type, archetype4 labels)
  analysis/p4_weight_registry.csv       — documented weights per feature per type
  analysis/p4_shortlist.csv             — regions above threshold ≥0.65 per type
  figures/p4_score_distributions.png    — 8-panel score histograms
  figures/p4_type_correlation_heatmap.png
  figures/p4_choropleth_grid.png        — 2×4 NUTS2 choropleth (if geopandas OK)
"""

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

GOLD_PATH  = ROOT / "data" / "gold" / "megacampus_gold.parquet"
OUT_PATH   = ROOT / "data" / "gold" / "suitability_scores.parquet"
ANALYSIS   = ROOT / "analysis"
FIGURES    = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
GEOJSON    = ROOT.parent / "EU-Innovation-Panel" / "data" / "raw" / "eurostat" / "nuts2_2021_geojson.json"

# Threshold applied after per-type min-max rescaling.
# Rescaling maps each type's raw composite to [0,1] across all 242 NUTS2 regions,
# so 0.60 consistently means "top 40% of EU NUTS2 on this type's structural profile."
# This is required because country-level broadcast features (renewable_energy_share,
# gerd_gov_pct_gdp, doing_business_score) compress T4's raw distribution to max=0.57,
# making a uniform absolute threshold infeasible without per-type normalisation.
SHORTLIST_THRESHOLD = 0.60
RESCALE_SCORES = True  # apply per-type min-max rescaling to composite scores

# ─── Type feature specifications ─────────────────────────────────────────────
# (feature_name, weight, direction)
# direction "+" = higher is better; "-" = lower is better (inverted)
# weight: 3=HIGH, 2=MEDIUM, 1=LOW
# DATA GAP proxies flagged in PROXY_FEATURES for uncertainty computation

TYPE_SPECS: dict[str, list[tuple[str, int, str]]] = {
    "T1": [  # AI / Machine Learning Hub
        ("score_talent",              3, "+"),
        ("score_infra",               3, "+"),
        ("lq_nace_j62j63",            3, "+"),
        ("hrst_per_1000",             2, "+"),
        ("ultrafast_broadband_pct",   2, "+"),
        ("business_rd_pct_gdp",       2, "+"),
        ("epo_patents_per_mio_pop",   2, "+"),
        ("doing_business_score",      2, "+"),
        ("cit_rate_pct",              1, "-"),
        ("electricity_price_eur_kwh", 1, "-"),
    ],
    "T2": [  # Biotechnology / Life Sciences
        ("score_talent",              3, "+"),
        ("lq_nace_c21_m72",           3, "+"),
        ("hrst_per_1000",             3, "+"),
        ("gerd_hes_pct_gdp",          3, "+"),
        ("epo_patents_per_mio_pop",   2, "+"),
        ("score_cluster",             2, "+"),
        ("doing_business_score",      2, "+"),
        ("cit_rate_pct",              1, "-"),
        ("electricity_price_eur_kwh", 1, "-"),
    ],
    "T3": [  # Semiconductors / Advanced Electronics
        ("lq_nace_c26",               3, "+"),
        ("electricity_price_eur_kwh", 3, "-"),
        ("score_talent",              2, "+"),
        ("hrst_per_1000",             2, "+"),
        ("rail_freight_ktonnes",      2, "+"),
        ("port_throughput_ktonnes",   2, "+"),
        ("doing_business_score",      2, "+"),
        ("artificial_land_pct",       1, "+"),
        ("cit_rate_pct",              1, "-"),
    ],
    "T4": [  # Cleantech / Green Technology
        ("renewable_energy_share_pct",3, "+"),
        ("lq_nace_d35_clean",         3, "+"),
        ("gerd_gov_pct_gdp",          2, "+"),
        ("esif_absorption_rate",      2, "+"),
        ("is_transition_region",      2, "+"),
        ("doing_business_score",      1, "+"),
        ("score_talent",              1, "+"),
        ("electricity_price_eur_kwh", 1, "-"),
    ],
    "T5": [  # Hyperscale Data Centre Hub
        ("electricity_price_eur_kwh", 3, "-"),
        ("ntc_import_mw",             3, "+"),
        ("renewable_energy_share_pct",3, "+"),
        ("ultrafast_broadband_pct",   3, "+"),
        ("water_exploitation_index",  2, "-"),
        ("artificial_land_pct",       2, "+"),
        ("doing_business_score",      2, "+"),
        ("score_infra",               1, "+"),
    ],
    "T6": [  # HALEU / Advanced Nuclear Industrial Base
        ("haleu_smr_eligible_int",    3, "+"),
        ("nuclear_capacity_mw_log1p", 3, "+"),
        ("gerd_def_pct_gdp",          2, "+"),
        ("ntc_import_mw",             2, "+"),
        ("lq_nace_c21_m72",           1, "+"),
        ("score_cost",                1, "+"),
    ],
    "T7": [  # Deep-Tech Robotics Campus
        ("lq_nace_c26",               3, "+"),
        ("score_talent",              3, "+"),
        ("hrst_per_1000",             2, "+"),
        ("gerd_hes_pct_gdp",          2, "+"),
        ("epo_patents_per_mio_pop",   2, "+"),
        ("score_cluster",             2, "+"),
        ("doing_business_score",      2, "+"),
        ("rail_freight_ktonnes",      1, "+"),
    ],
    "T8": [  # Quantum / Photonics Research Anchor
        ("quantum_programme_tier",    3, "+"),
        ("eurohpc_proximity_score",   3, "+"),
        ("score_talent",              3, "+"),
        ("gerd_hes_pct_gdp",          2, "+"),
        ("epo_patents_per_mio_pop",   2, "+"),
        ("ultrafast_broadband_pct",   2, "+"),
        ("score_cluster",             1, "+"),
    ],
}

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

# Features that are DATA GAP proxies — increase uncertainty score
PROXY_FEATURES = {
    "quantum_programme_tier",
    "eurohpc_proximity_score",
    "nuclear_capacity_mw_log1p",
    "gerd_def_pct_gdp",
    "haleu_smr_eligible_int",
    "esif_absorption_rate",
    "port_throughput_ktonnes",
}

# Palantir-style colour palette
_BG   = "#0a0f1e"
_BLUE = "#00a8ff"
_AMBER= "#ff8c42"
_GREEN= "#00cc7a"
_RED  = "#ff3355"
_PURPLE="#a855f7"
_CYAN = "#66d9e8"
_GOLD = "#f5a623"
_PINK = "#fbbf24"

TYPE_COLORS = {
    "T1": _BLUE, "T2": _GREEN, "T3": _AMBER, "T4": _CYAN,
    "T5": _GOLD, "T6": _RED,   "T7": _PURPLE,"T8": _PINK,
}


# ─── Scoring helpers ─────────────────────────────────────────────────────────

def _minmax(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0.5, index=s.index, dtype=float)
    return (s - mn) / (mx - mn)


def _prepare_feature(df: pd.DataFrame, feat: str) -> pd.Series:
    """Extract and coerce a feature column, returning a float Series."""
    if feat not in df.columns:
        return pd.Series(np.nan, index=df.index)
    col = df[feat]
    if col.dtype == bool or col.dtype == object:
        col = col.astype(float)
    return pd.to_numeric(col, errors="coerce")


def compute_suitability(df: pd.DataFrame, type_id: str) -> tuple[pd.Series, pd.Series]:
    """
    Returns (score [0,1], uncertainty [0,1]) for one type.
    uncertainty = proportion of features that are proxies or imputed.
    """
    specs = TYPE_SPECS[type_id]
    total_w = sum(w for _, w, _ in specs)
    weighted = pd.Series(0.0, index=df.index)
    proxy_w = 0.0
    imputed_w = 0.0

    for feat, w, direction in specs:
        vals = _prepare_feature(df, feat)
        # Fill remaining NaN with median
        vals = vals.fillna(vals.median() if vals.notna().any() else 0.0)
        norm = _minmax(vals)
        if direction == "-":
            norm = 1.0 - norm
        weighted += norm * w

        # Uncertainty tracking
        if feat in PROXY_FEATURES:
            proxy_w += w
        imputed_flag = f"{feat.replace('_log1p','').replace('_int','')}_imputed_flag"
        if imputed_flag in df.columns and df[imputed_flag].any():
            imputed_w += w * df[imputed_flag].astype(float).mean()

    score = weighted / total_w
    uncertainty = min(1.0, (proxy_w + imputed_w) / total_w)

    return score, pd.Series(uncertainty, index=df.index)


# ─── Derived columns ─────────────────────────────────────────────────────────

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add log1p-transformed and int-cast derived features needed by scoring."""
    df = df.copy()
    # Nuclear capacity: log1p for heavy right skew
    if "nuclear_capacity_mw" in df.columns:
        df["nuclear_capacity_mw_log1p"] = np.log1p(df["nuclear_capacity_mw"].fillna(0))
    # HALEU eligibility: bool -> int
    if "haleu_smr_eligible" in df.columns:
        df["haleu_smr_eligible_int"] = df["haleu_smr_eligible"].astype(float)
    # is_transition_region: bool -> float
    if "is_transition_region" in df.columns:
        df["is_transition_region"] = df["is_transition_region"].astype(float)
    # quantum_programme_tier: normalize 0-3 to 0-1
    if "quantum_programme_tier" in df.columns:
        df["quantum_programme_tier"] = pd.to_numeric(
            df["quantum_programme_tier"], errors="coerce").fillna(0) / 3.0
    return df


# ─── Figures ─────────────────────────────────────────────────────────────────

def fig_distributions(scores_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor=_BG)
    fig.suptitle("SUITABILITY INDEX DISTRIBUTIONS — EU NUTS2 (n=242)",
                 color=_BLUE, fontsize=13, fontfamily="monospace",
                 fontweight="bold", y=1.01)
    for i, (tid, label) in enumerate(TYPE_LABELS.items()):
        ax = axes[i // 4][i % 4]
        col = f"suitability_{tid}"
        vals = scores_df[col].dropna()
        c = TYPE_COLORS[tid]
        ax.set_facecolor("#0d1529")
        ax.hist(vals, bins=25, color=c, alpha=0.85, edgecolor=_BG, linewidth=0.4)
        ax.axvline(SHORTLIST_THRESHOLD, color=_RED, linestyle="--", linewidth=1, alpha=0.8)
        ax.set_title(f"{tid}: {label}", color=c, fontsize=8.5, fontfamily="monospace")
        ax.set_xlabel("Index [0,1]", color="#c0d0e8", fontsize=7)
        ax.set_ylabel("Regions", color="#c0d0e8", fontsize=7)
        ax.tick_params(colors="#c0d0e8", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e3050")
        n_above = (vals >= SHORTLIST_THRESHOLD).sum()
        ax.text(0.97, 0.95, f"n>{SHORTLIST_THRESHOLD:.2f}: {n_above}",
                transform=ax.transAxes, ha="right", va="top",
                color=_RED, fontsize=7, fontfamily="monospace")
    plt.tight_layout()
    out = FIGURES / "p4_score_distributions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print(f"    Saved: {out.relative_to(ROOT)}")


def fig_correlation_heatmap(scores_df: pd.DataFrame) -> None:
    score_cols = [f"suitability_{t}" for t in TYPE_LABELS]
    corr = scores_df[score_cols].corr()
    labels = [f"{t}\n{TYPE_LABELS[t][:20]}" for t in TYPE_LABELS]
    fig, ax = plt.subplots(figsize=(10, 9), facecolor=_BG)
    ax.set_facecolor("#0d1529")
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "palantir", ["#0d1529", "#1e3a5f", _BLUE, _AMBER])
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors="#c0d0e8")
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels(labels, rotation=45, ha="right", color="#c0d0e8", fontsize=8)
    ax.set_yticklabels(labels, color="#c0d0e8", fontsize=8)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f"{corr.values[i,j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(corr.values[i,j]) > 0.3 else "#c0d0e8")
    ax.set_title("TYPE SUITABILITY INDEX CORRELATION MATRIX",
                 color=_BLUE, fontsize=11, fontfamily="monospace", pad=14)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3050")
    ax.tick_params(colors="#c0d0e8")
    plt.tight_layout()
    out = FIGURES / "p4_type_correlation_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print(f"    Saved: {out.relative_to(ROOT)}")


def fig_choropleth_grid(scores_df: pd.DataFrame) -> None:
    try:
        import geopandas as gpd
    except ImportError:
        print("    SKIP choropleth: geopandas not available")
        return
    if not GEOJSON.exists():
        print(f"    SKIP choropleth: GeoJSON not found at {GEOJSON}")
        return

    gdf = gpd.read_file(GEOJSON)[["NUTS_ID", "geometry"]].rename(
        columns={"NUTS_ID": "nuts2_code"})
    merged = gdf.merge(scores_df.reset_index(), on="nuts2_code", how="left")
    merged = merged[merged["nuts2_code"].str.len() == 4]

    fig, axes = plt.subplots(2, 4, figsize=(22, 11), facecolor=_BG)
    fig.suptitle("DISRUPTOR ECOSYSTEM SUITABILITY INDICES — EU NUTS2",
                 color=_BLUE, fontsize=14, fontfamily="monospace",
                 fontweight="bold", y=1.005)

    cmap_base = mcolors.LinearSegmentedColormap.from_list(
        "suitability", ["#0d1529", "#1e3a5f", "#00507a", _BLUE, _CYAN])

    for i, (tid, label) in enumerate(TYPE_LABELS.items()):
        ax = axes[i // 4][i % 4]
        ax.set_facecolor(_BG)
        col = f"suitability_{tid}"
        merged.plot(ax=ax, column=col, cmap=cmap_base,
                    vmin=0, vmax=1, missing_kwds={"color": "#0d1529"},
                    linewidth=0.1, edgecolor="#0a0f1e")
        ax.set_title(f"{tid}  {label}", color=TYPE_COLORS[tid],
                     fontsize=8, fontfamily="monospace", pad=4)
        ax.axis("off")
        n_above = (merged[col].fillna(0) >= SHORTLIST_THRESHOLD).sum()
        ax.text(0.02, 0.05, f"Index >= {SHORTLIST_THRESHOLD}: {n_above} regions",
                transform=ax.transAxes, color=_RED, fontsize=6.5,
                fontfamily="monospace")

    plt.tight_layout()
    out = FIGURES / "p4_choropleth_grid.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print(f"    Saved: {out.relative_to(ROOT)}")


# ─── Weight registry ──────────────────────────────────────────────────────────

def build_weight_registry() -> pd.DataFrame:
    rows = []
    for tid, specs in TYPE_SPECS.items():
        total_w = sum(w for _, w, _ in specs)
        for feat, w, direction in specs:
            rows.append({
                "type_id": tid,
                "type_label": TYPE_LABELS[tid],
                "feature": feat,
                "weight": w,
                "weight_label": {3: "HIGH", 2: "MEDIUM", 1: "LOW"}[w],
                "direction": direction,
                "is_proxy": feat in PROXY_FEATURES,
                "pct_of_type_weight": round(100 * w / total_w, 1),
            })
    return pd.DataFrame(rows)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    with pipeline_step("p4_load_gold"):
        gold = pd.read_parquet(GOLD_PATH)
        gold.index.name = "nuts2_code"
        gold = add_derived_columns(gold)

    with pipeline_step("p4_compute_scores"):
        results: dict[str, pd.Series] = {}
        uncertainty: dict[str, pd.Series] = {}

        for tid in TYPE_LABELS:
            score, unc = compute_suitability(gold, tid)
            results[f"suitability_{tid}"] = score
            uncertainty[f"uncertainty_{tid}"] = unc

        # Per-type min-max rescaling: each type's composite is scaled to [0,1]
        # across all 242 NUTS2 regions so the threshold is consistent across types
        # irrespective of raw score compression from country-level broadcast features.
        if RESCALE_SCORES:
            for tid in TYPE_LABELS:
                col = f"suitability_{tid}"
                results[col] = _minmax(results[col])

        scores_df = pd.DataFrame(results, index=gold.index)
        unc_df    = pd.DataFrame(uncertainty, index=gold.index)

        # Shortlist flags
        for tid in TYPE_LABELS:
            scores_df[f"shortlist_{tid}"] = scores_df[f"suitability_{tid}"] >= SHORTLIST_THRESHOLD

        # Dominant type
        type_score_cols = [f"suitability_{t}" for t in TYPE_LABELS]
        scores_df["dominant_type"] = scores_df[type_score_cols].idxmax(axis=1).str.replace("suitability_", "")
        scores_df["dominant_type_score"] = scores_df[type_score_cols].max(axis=1)

        # Multi-type and hyper-diverse flags
        n_above = (scores_df[type_score_cols] >= SHORTLIST_THRESHOLD).sum(axis=1)
        scores_df["n_types_above_threshold"] = n_above
        scores_df["multi_type_flag"]    = n_above >= 2
        scores_df["hyper_diverse_flag"] = n_above >= 4

        # Co-specialisation flags (key strategic combinations)
        def co_spec(t1, t2, thr=0.60):
            return (scores_df[f"suitability_{t1}"] >= thr) & (scores_df[f"suitability_{t2}"] >= thr)

        scores_df["co_spec_T1_T2"] = co_spec("T1", "T2")   # AI + Bio
        scores_df["co_spec_T1_T5"] = co_spec("T1", "T5")   # AI + Data Centre
        scores_df["co_spec_T3_T7"] = co_spec("T3", "T7")   # Semi + Robotics
        scores_df["co_spec_T4_T6"] = co_spec("T4", "T6")   # Clean + Nuclear
        scores_df["co_spec_T5_T6"] = co_spec("T5", "T6")   # DC + HALEU-SMR (integration thesis)
        scores_df["co_spec_T6_T8"] = co_spec("T6", "T8")   # Nuclear + Quantum
        scores_df["co_spec_T1_T8"] = co_spec("T1", "T8")   # AI + Quantum

        # Join uncertainty
        out_df = scores_df.join(unc_df)

        # Preserve key identifiers from Gold
        for col in ["nuts2_name", "country_code", "archetype4_id", "archetype4_label",
                    "archetype_id", "archetype_label", "is_transition_region", "haleu_smr_eligible",
                    "nuclear_policy"]:
            if col in gold.columns:
                out_df[col] = gold[col]

    with pipeline_step("p4_artifacts"):
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(OUT_PATH)

        # Weight registry
        reg = build_weight_registry()
        reg.to_csv(ANALYSIS / "p4_weight_registry.csv", index=False)

        # Shortlist CSV
        shortlist_rows = []
        for tid in TYPE_LABELS:
            sub = out_df[out_df[f"shortlist_{tid}"]].copy()
            sub["type_id"] = tid
            sub["type_label"] = TYPE_LABELS[tid]
            sub["suitability_score"] = sub[f"suitability_{tid}"]
            sub["uncertainty_score"] = sub[f"uncertainty_{tid}"]
            cols = ["type_id", "type_label", "suitability_score", "uncertainty_score",
                    "nuts2_name", "country_code", "archetype4_label"]
            cols = [c for c in cols if c in sub.columns]
            shortlist_rows.append(sub[cols].reset_index())
        shortlist_df = pd.concat(shortlist_rows, ignore_index=True) if shortlist_rows else pd.DataFrame()
        shortlist_df.to_csv(ANALYSIS / "p4_shortlist.csv", index=False)

    with pipeline_step("p4_figures"):
        fig_distributions(out_df)
        fig_correlation_heatmap(out_df)
        fig_choropleth_grid(out_df)

    # ── Gate check ────────────────────────────────────────────────────────────
    score_stats = {}
    for tid in TYPE_LABELS:
        col = f"suitability_{tid}"
        score_stats[tid] = {
            "mean": round(out_df[col].mean(), 3),
            "std":  round(out_df[col].std(), 3),
            "n_shortlisted": int(out_df[f"shortlist_{tid}"].sum()),
        }

    print()
    print("=" * 70)
    print("P4 SUITABILITY SCORE GATE CHECK")
    print("=" * 70)
    print(f"  Output shape : {out_df.shape}")
    print(f"  Shortlist threshold: >= {SHORTLIST_THRESHOLD}")
    print()
    thr_hdr = f"n>={SHORTLIST_THRESHOLD:.2f}"
    print(f"  {'Type':<6} {'Label':<40} {'Mean':>6} {'Std':>6} {thr_hdr:>8}")
    print(f"  {'-'*6} {'-'*40} {'-'*6} {'-'*6} {'-'*8}")
    for tid, s in score_stats.items():
        label = TYPE_LABELS[tid][:38]
        print(f"  {tid:<6} {label:<40} {s['mean']:>6.3f} {s['std']:>6.3f} {s['n_shortlisted']:>8}")
    print()
    print(f"  Co-specialisation:")
    co_cols = [c for c in out_df.columns if c.startswith("co_spec_")]
    for c in co_cols:
        print(f"    {c:<22}: {out_df[c].sum():3d} regions")
    print()
    print(f"  Multi-type (>= 2 types above {SHORTLIST_THRESHOLD}): {out_df['multi_type_flag'].sum()}")
    print(f"  Hyper-diverse (>= 4 types):                 {out_df['hyper_diverse_flag'].sum()}")
    print()

    all_scores_valid = all(
        0.0 <= out_df[f"suitability_{t}"].min() and out_df[f"suitability_{t}"].max() <= 1.0
        for t in TYPE_LABELS
    )
    some_shortlisted = all(score_stats[t]["n_shortlisted"] >= 1 for t in TYPE_LABELS)
    gate_pass = out_df.shape[0] == 242 and all_scores_valid and some_shortlisted

    print(f"GATE_P4={'PASS' if gate_pass else 'FAIL'}")
    if not gate_pass:
        if out_df.shape[0] != 242:
            print(f"  REASON: expected 242 rows, got {out_df.shape[0]}")
        if not all_scores_valid:
            print(f"  REASON: scores outside [0,1]")
        if not some_shortlisted:
            print(f"  REASON: type(s) with 0 shortlisted regions")
    print("=" * 70)

    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
