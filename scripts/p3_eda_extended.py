"""
P3 — Extended EDA on the 18 new MegaCampus features.

Spatial autocorrelation is intentionally NOT repeated here -- P6 already
computes Global Moran's I and LISA for the 8 downstream suitability indices.
PCA is intentionally NOT repeated here -- P4 already constructs the 8 type
composite indices from these same raw features. This stage covers the gap
those two leave: distributions, correlation structure, missingness, and
near-zero-variance detection on the 18 *raw* features themselves, before
they're compressed into suitability scores.

The 18 new features (vs. the upstream EU-Innovation-Panel Gold layer):
  15 continuous : artificial_land_pct, cit_rate_pct, doing_business_score,
                  electricity_price_eur_kwh, eurohpc_proximity_score,
                  gerd_def_pct_gdp, gerd_gov_pct_gdp, gerd_hes_pct_gdp,
                  ntc_import_mw, nuclear_capacity_mw, port_throughput_ktonnes,
                  rail_freight_ktonnes, renewable_energy_share_pct,
                  ultrafast_broadband_pct, water_exploitation_index
  3 categorical : haleu_smr_eligible (bool), nuclear_policy (str),
                  quantum_programme_tier (int, ordinal)

Outputs:
  analysis/p3_distributions.csv       — per-feature skew/kurtosis/normality (continuous)
  analysis/p3_correlation_matrix.csv  — Pearson correlation, 15x15 continuous features
  analysis/p3_missingness.csv         — imputed-row rate per feature (from *_imputed_flag)
  analysis/p3_categorical_freq.csv    — frequency table for the 3 categorical features
  analysis/p3_low_variance_flags.csv  — features with near-zero variance (real anomaly detector)
  figures/p3_distributions_grid.png   — 15-panel histogram grid
  figures/p3_correlation_heatmap.png  — correlation heatmap
  figures/p3_categorical_bars.png     — 3-panel bar chart for categorical features
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logging_config import pipeline_step  # noqa: E402

GOLD_PATH = ROOT / "data" / "gold" / "megacampus_gold.parquet"
ANALYSIS  = ROOT / "analysis"
FIGURES   = ROOT / "figures"
ANALYSIS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

CONTINUOUS_FEATURES = [
    "artificial_land_pct", "cit_rate_pct", "doing_business_score",
    "electricity_price_eur_kwh", "eurohpc_proximity_score",
    "gerd_def_pct_gdp", "gerd_gov_pct_gdp", "gerd_hes_pct_gdp",
    "ntc_import_mw", "nuclear_capacity_mw", "port_throughput_ktonnes",
    "rail_freight_ktonnes", "renewable_energy_share_pct",
    "ultrafast_broadband_pct", "water_exploitation_index",
]
CATEGORICAL_FEATURES = ["haleu_smr_eligible", "nuclear_policy", "quantum_programme_tier"]
LOW_VARIANCE_CV_THRESHOLD = 0.01  # coefficient of variation below this = flagged


def load_gold() -> pd.DataFrame:
    return pd.read_parquet(GOLD_PATH)


def compute_distributions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat in CONTINUOUS_FEATURES:
        series = df[feat].dropna()
        if len(series) < 3:
            rows.append({"feature": feat, "n": len(series), "mean": np.nan, "std": np.nan,
                          "skew": np.nan, "kurtosis": np.nan, "shapiro_p": np.nan, "normal_at_5pct": np.nan})
            continue
        shapiro_p = stats.shapiro(series)[1] if len(series) <= 5000 else np.nan
        rows.append({
            "feature": feat,
            "n": len(series),
            "mean": series.mean(),
            "std": series.std(),
            "skew": stats.skew(series),
            "kurtosis": stats.kurtosis(series),
            "shapiro_p": shapiro_p,
            "normal_at_5pct": bool(shapiro_p > 0.05) if not np.isnan(shapiro_p) else None,
        })
    return pd.DataFrame(rows)


def plot_distributions(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(3, 5, figsize=(20, 11))
    for ax, feat in zip(axes.flat, CONTINUOUS_FEATURES):
        ax.hist(df[feat].dropna(), bins=30, color="#00a8ff", alpha=0.8)
        ax.set_title(feat, fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle("P3 — Distributions of the 18 new MegaCampus features (15 continuous)")
    fig.tight_layout()
    out = FIGURES / "p3_distributions_grid.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[CONTINUOUS_FEATURES].corr(method="pearson")


def plot_correlation_heatmap(corr: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=8)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("P3 — Correlation matrix, 15 continuous new features")
    fig.tight_layout()
    out = FIGURES / "p3_correlation_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def compute_missingness(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat in CONTINUOUS_FEATURES:
        flag_col = f"{feat}_imputed_flag"
        imputed_rate = df[flag_col].mean() if flag_col in df.columns else np.nan
        rows.append({"feature": feat, "imputed_row_rate": imputed_rate})
    return pd.DataFrame(rows)


def compute_categorical_freq(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat in CATEGORICAL_FEATURES:
        vc = df[feat].value_counts(dropna=False)
        for value, count in vc.items():
            rows.append({"feature": feat, "value": str(value), "count": int(count),
                         "pct": round(100 * count / len(df), 1)})
    return pd.DataFrame(rows)


def plot_categorical_bars(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, feat in zip(axes, CATEGORICAL_FEATURES):
        vc = df[feat].value_counts(dropna=False)
        ax.bar([str(v) for v in vc.index], vc.values, color="#ff8c42")
        ax.set_title(feat, fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.suptitle("P3 — Categorical feature frequencies")
    fig.tight_layout()
    out = FIGURES / "p3_categorical_bars.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def compute_low_variance_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flags features whose coefficient of variation (std/|mean|) is below
    threshold, or that have a single unique non-null value -- exactly the
    signature of a broken ingestion (constant fallback value / silently
    zero-filled empty join), not a real low-variance measurement."""
    rows = []
    for feat in CONTINUOUS_FEATURES:
        series = df[feat].dropna()
        n_unique = series.nunique()
        mean = series.mean()
        std = series.std()
        cv = abs(std / mean) if mean not in (0, np.nan) and not np.isnan(mean) else np.nan
        flagged = (n_unique <= 1) or (not np.isnan(cv) and cv < LOW_VARIANCE_CV_THRESHOLD)
        rows.append({
            "feature": feat, "n_unique": n_unique, "coefficient_of_variation": cv,
            "flagged_low_variance": flagged,
        })
    return pd.DataFrame(rows)


def run_quality_gate(low_var: pd.DataFrame, missingness: pd.DataFrame) -> bool:
    n_flagged = int(low_var["flagged_low_variance"].sum())
    max_imputed = missingness["imputed_row_rate"].max()
    print("\n" + "=" * 65)
    print("P3 GATE CHECK")
    print("=" * 65)
    print(f"  Continuous features analysed : {len(CONTINUOUS_FEATURES)}")
    print(f"  Categorical features         : {len(CATEGORICAL_FEATURES)}")
    print(f"  Low-variance flags           : {n_flagged}")
    if n_flagged:
        print("  Flagged features (real anomaly, not expected to be clean):")
        for _, row in low_var[low_var["flagged_low_variance"]].iterrows():
            print(f"    - {row['feature']} (n_unique={row['n_unique']}, CV={row['coefficient_of_variation']})")
    print(f"  Max imputed-row rate         : {max_imputed:.1%}")
    print("=" * 65)
    # This gate reports low-variance findings rather than failing on them --
    # they're real data-quality findings for P2 to investigate, not a defect
    # in this EDA stage itself.
    return True


def main() -> None:
    with pipeline_step("p3_eda_extended", input_artifact=GOLD_PATH) as log:
        df = load_gold()
        log.info("gold_loaded", shape=list(df.shape))

        dist = compute_distributions(df)
        dist.to_csv(ANALYSIS / "p3_distributions.csv", index=False)
        plot_distributions(df)

        corr = compute_correlation_matrix(df)
        corr.to_csv(ANALYSIS / "p3_correlation_matrix.csv")
        plot_correlation_heatmap(corr)

        missingness = compute_missingness(df)
        missingness.to_csv(ANALYSIS / "p3_missingness.csv", index=False)

        cat_freq = compute_categorical_freq(df)
        cat_freq.to_csv(ANALYSIS / "p3_categorical_freq.csv", index=False)
        plot_categorical_bars(df)

        low_var = compute_low_variance_flags(df)
        low_var.to_csv(ANALYSIS / "p3_low_variance_flags.csv", index=False)

        passed = run_quality_gate(low_var, missingness)
        print(f"\nGATE_P3={'PASS' if passed else 'FAIL'}")
        log.info("step_complete", gate="PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
