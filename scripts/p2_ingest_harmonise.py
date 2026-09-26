"""
P2 — Ingestion, Harmonisation, k=4 Clustering, Silver + Gold build.

Loads the EU-Innovation-Panel Gold layer as the base, ingests all 16 new
datasets from data/raw/, harmonises to NUTS2 (2021), runs k=4 k-means on
the 4 upstream dimension scores (intentional interpretability override from k=2),
and writes data/silver/ and data/gold/ parquets.

Outputs:
  data/silver/megacampus_silver.parquet   — 242 × ~50 harmonised features
  data/gold/megacampus_gold.parquet       — Silver + k=4 clusters + derived flags
  analysis/p2_ingestion_report.json       — per-feature coverage and decisions
  analysis/p2_clustering_diagnostics.csv  — silhouette, ARI, centroid profiles
  analysis/p2_year_alignment.csv          — reference year per new feature
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logging_config import pipeline_step  # noqa: E402

UPSTREAM_GOLD = ROOT.parent / "EU-Innovation-Panel" / "data" / "gold" / "region_profiles_gold.parquet"
RAW = ROOT / "data" / "raw"
SILVER_PATH = ROOT / "data" / "silver" / "megacampus_silver.parquet"
GOLD_PATH = ROOT / "data" / "gold" / "megacampus_gold.parquet"
ANALYSIS = ROOT / "analysis"

SCORE_COLS = ["score_cost", "score_talent", "score_infra", "score_cluster"]
EU27 = ["AT","BE","BG","CY","CZ","DE","DK","EE","ES","FI",
        "FR","GR","HR","HU","IE","IT","LT","LU","LV","MT",
        "NL","PL","PT","RO","SE","SI","SK"]

# ─── Generic Eurostat SDMX-CSV reader ────────────────────────────────────────

def _read_sdmx(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _pick_year(df: pd.DataFrame, geo_col: str, time_col: str, val_col: str,
               min_coverage: float = 0.5) -> tuple[str | None, pd.Series]:
    """Return (reference_year, Series indexed by geo) for best-covered year."""
    df = df[[geo_col, time_col, val_col]].dropna(subset=[val_col])
    if df.empty:
        return None, pd.Series(dtype=float)
    n_total = df[geo_col].nunique()
    for year in sorted(df[time_col].unique(), reverse=True):
        sub = df[df[time_col] == year]
        if len(sub) / max(n_total, 1) >= min_coverage:
            return str(year), sub.set_index(geo_col)[val_col]
    # fallback: latest year regardless
    year = sorted(df[time_col].unique())[-1]
    sub = df[df[time_col] == year]
    return str(year), sub.set_index(geo_col)[val_col]


# ─── Feature parsers ─────────────────────────────────────────────────────────

def parse_electricity_price() -> tuple[str, pd.Series]:
    """D01: nrg_pc_205 — industrial electricity price EUR/kWh, country-level."""
    df = _read_sdmx(RAW / "eurostat" / "nrg_pc_205.csv")
    # Standardise column names
    df.columns = [c.lower().strip() for c in df.columns]
    # Identify geo column
    geo_col = next((c for c in df.columns if c == "geo"), None)
    if geo_col is None:
        geo_col = next((c for c in df.columns if "geo" in c), df.columns[0])
    time_col = next((c for c in df.columns if "time" in c), None)
    val_col = "obs_value"
    if val_col not in df.columns:
        val_col = next((c for c in df.columns if "obs" in c and "value" in c.lower()), df.columns[-3])
    # Filter: KWH unit, EUR currency, tax included (I_TAX) or ex-tax (X_TAX)
    for col in ["unit", "currency", "tax"]:
        if col in df.columns:
            if col == "unit":
                df = df[df[col].str.upper().str.contains("KWH", na=False)]
            elif col == "currency":
                df = df[df[col].str.upper() == "EUR"]
            elif col == "tax":
                df = df[df[col].str.upper().isin(["I_TAX", "X_TAX"])]
    # Filter to EU27 country-level codes only
    df = df[df[geo_col].str.len() == 2]
    df = df[df[geo_col].isin(EU27)]
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    # Average across consumer bands for each geo × year
    agg = df.groupby([geo_col, time_col])[val_col].mean().reset_index()
    year, series = _pick_year(agg, geo_col, time_col, val_col)
    series.name = "electricity_price_eur_kwh"
    return year or "2022", series


def parse_renewable_share() -> tuple[str, pd.Series]:
    """D02: nrg_ind_ren — renewable share of gross final energy consumption (%)."""
    df = _read_sdmx(RAW / "eurostat" / "nrg_ind_ren.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    geo_col = next((c for c in df.columns if c == "geo"), df.columns[0])
    time_col = next((c for c in df.columns if "time" in c), None)
    val_col = "obs_value"
    if val_col not in df.columns:
        val_col = next((c for c in df.columns if "obs" in c), df.columns[-3])
    # Filter: REN balance, TOTAL siec, PC unit
    for col, val in [("nrg_bal", "REN"), ("siec", "TOTAL"), ("unit", "PC")]:
        if col in df.columns:
            df = df[df[col].str.upper() == val]
    df = df[df[geo_col].str.len() == 2]
    df = df[df[geo_col].isin(EU27)]
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    year, series = _pick_year(df, geo_col, time_col, val_col)
    series.name = "renewable_energy_share_pct"
    return year or "2022", series


def parse_rail_freight() -> tuple[str, pd.Series]:
    """D05: tran_r_rago -- national+international railway goods transport by
    loading/unloading NUTS2 region, tonnes.

    NOTE (2026-09-26 fix): this previously read tran_r_rapa.csv, which is
    Eurostat's *passengers* dataset ("rapa" = embarking/disembarking
    passengers), not freight -- a one-letter dataset-code mixup with the
    correct tran_r_rago ("rago" = goods) series. tran_r_rapa's `geo` column
    is also only ever 2-char country codes in this extract, which combined
    with the wrong dataset meant every row was filtered out by the old
    NUTS2-length check, silently producing an all-zero feature via
    reindex().fillna(0.0) at the call site. tran_r_rago has genuine NUTS2
    granularity, but as bilateral (loading-region, unloading-region) flow
    data: the `geo` column there is a reporting-country dimension, and the
    actual NUTS2 detail lives in c_load/c_unload. A region's total rail
    freight is the sum of tonnage where it appears as either the loading or
    the unloading region.
    """
    df = _read_sdmx(RAW / "eurostat" / "tran_r_rago.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    time_col = next((c for c in df.columns if "time" in c), None)
    val_col = "obs_value"
    if val_col not in df.columns:
        val_col = next((c for c in df.columns if "obs" in c), df.columns[-3])
    if "unit" in df.columns:
        df = df[df["unit"].str.upper() == "T"]
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")

    loaded = df[df["c_load"].str.len() == 4].rename(columns={"c_load": "nuts2"})
    unloaded = df[df["c_unload"].str.len() == 4].rename(columns={"c_unload": "nuts2"})
    combined = pd.concat([
        loaded[["nuts2", time_col, val_col]],
        unloaded[["nuts2", time_col, val_col]],
    ], ignore_index=True)

    agg = combined.groupby(["nuts2", time_col])[val_col].sum().reset_index()
    year, series = _pick_year(agg, "nuts2", time_col, val_col)
    series.name = "rail_freight_ktonnes"
    return year or "2020", series


def parse_port_throughput() -> tuple[str, pd.Series]:
    """D06: mar_go_aa — port throughput aggregated to country (thousand tonnes).

    NOTE (2026-09-26 fix): mar_go_aa.csv has no `geo` column -- the old code's
    `next((c for c in df.columns if c == "geo"), df.columns[0])` fallback
    silently used `df.columns[0]` ("dataflow", a constant string like
    "ESTAT:MAR_GO_AA(1.0)" for every row), whose first 2 characters ("ES")
    coincidentally matched Spain's real ISO code. That made every country's
    port tonnage collapse into one mislabeled "ES" total, leaving 26 of 27
    countries with no matching broadcast target -- which is exactly why the
    Gold-layer feature ended up as a single EU-median-imputed constant for
    92% of regions (see scripts/p3_eda_extended.py's low-variance flag).
    The actual geo dimension is `rep_mar` (reporting maritime area); country
    totals are the rows where it's already a 2-char country code (as
    opposed to individual port codes), with `direct == "TOTAL"` to avoid
    double-counting inbound+outbound separately.
    """
    df = _read_sdmx(RAW / "eurostat" / "mar_go_aa.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    time_col = next((c for c in df.columns if "time" in c), None)
    val_col = "obs_value"
    if val_col not in df.columns:
        val_col = next((c for c in df.columns if "obs" in c), df.columns[-3])
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")

    df = df[df["rep_mar"].str.len() == 2]
    df["country_code"] = df["rep_mar"].str.upper()
    if "direct" in df.columns:
        df = df[df["direct"] == "TOTAL"]
    df = df[df["country_code"].isin(EU27)]
    agg = df.groupby(["country_code", time_col])[val_col].sum().reset_index()
    year, series = _pick_year(agg, "country_code", time_col, val_col)
    series.name = "port_throughput_ktonnes_country"
    return year or "2022", series


def parse_gerd_by_sector() -> tuple[str, dict[str, pd.Series]]:
    """D07: rd_e_gerdsc — GERD by sector (GOV, HES, DEF) as % GDP, country-level."""
    df = _read_sdmx(RAW / "eurostat" / "rd_e_gerdsc.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    geo_col = next((c for c in df.columns if c == "geo"), df.columns[0])
    time_col = next((c for c in df.columns if "time" in c), None)
    val_col = "obs_value"
    if val_col not in df.columns:
        val_col = next((c for c in df.columns if "obs" in c), df.columns[-3])
    # Unit: PC_GDP (percent of GDP)
    sect_col = next((c for c in df.columns if "sect" in c), None)
    unit_col = next((c for c in df.columns if c == "unit"), None)
    if unit_col and "pc_gdp" in df[unit_col].str.lower().values:
        df = df[df[unit_col].str.upper() == "PC_GDP"]
    df = df[df[geo_col].str.len() == 2]
    df = df[df[geo_col].isin(EU27)]
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    result = {}
    year_used = "2021"
    for sect, name in [("GOV", "gerd_gov_pct_gdp"), ("HES", "gerd_hes_pct_gdp"), ("DEF", "gerd_def_pct_gdp")]:
        if sect_col and sect in df[sect_col].str.upper().values:
            sub = df[df[sect_col].str.upper() == sect]
        else:
            sub = df  # fallback: use all
        year_used, series = _pick_year(sub, geo_col, time_col, val_col)
        series.name = name
        result[name] = series
    return year_used or "2021", result


def parse_ultrafast_broadband() -> tuple[str, pd.Series]:
    """D11: isoc_r_broad_h — ultra-fast broadband (>100 Mbps) coverage, NUTS2."""
    df = _read_sdmx(RAW / "eurostat" / "isoc_r_broad_h.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    geo_col = next((c for c in df.columns if c == "geo"), df.columns[0])
    time_col = next((c for c in df.columns if "time" in c), None)
    val_col = "obs_value"
    if val_col not in df.columns:
        val_col = next((c for c in df.columns if "obs" in c), df.columns[-3])
    # Filter for PC_HH unit and ultra-fast bandwidth type
    for col in ["unit"]:
        if col in df.columns:
            df = df[df[col].str.upper().str.contains("PC", na=False)]
    # Filter bandwidth: look for NGA100, FA100, or ≥100Mbps labels
    bband_col = next((c for c in df.columns if "broad" in c and c != geo_col), None)
    if bband_col:
        mask = (df[bband_col].str.upper().str.contains("100", na=False) |
                df[bband_col].str.upper().str.contains("NGA", na=False) |
                df[bband_col].str.upper().str.contains("TOTAL", na=False))
        sub = df[mask]
        if sub.empty:
            sub = df  # fallback: all broadband
    else:
        sub = df
    sub = sub[sub[geo_col].str.len() == 4]
    sub[val_col] = pd.to_numeric(sub[val_col], errors="coerce")
    agg = sub.groupby([geo_col, time_col])[val_col].mean().reset_index()
    year, series = _pick_year(agg, geo_col, time_col, val_col)
    series.name = "ultrafast_broadband_pct"
    return year or "2022", series


def parse_land_cover() -> tuple[str, pd.Series]:
    """D12: lan_lcv_ovw — artificial/industrial land share (%), NUTS2."""
    df = _read_sdmx(RAW / "eurostat" / "lan_lcv_ovw.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    geo_col = next((c for c in df.columns if c == "geo"), df.columns[0])
    time_col = next((c for c in df.columns if "time" in c), None)
    val_col = "obs_value"
    if val_col not in df.columns:
        val_col = next((c for c in df.columns if "obs" in c), df.columns[-3])
    # Filter for artificial surfaces (CLC class 1xx or lcv_type='ART' or unit='PC')
    lcv_col = next((c for c in df.columns if "lcv" in c or "land" in c), None)
    if lcv_col:
        art_mask = df[lcv_col].str.upper().str.contains("ART|ARTIF|URB|1[0-9][0-9]", na=False, regex=True)
        sub = df[art_mask] if art_mask.any() else df
    else:
        sub = df
    # Prefer PC (percent) unit
    unit_col = next((c for c in df.columns if c == "unit"), None)
    if unit_col and "PC" in sub[unit_col].str.upper().values:
        sub = sub[sub[unit_col].str.upper() == "PC"]
    sub = sub[sub[geo_col].str.len() == 4]
    sub[val_col] = pd.to_numeric(sub[val_col], errors="coerce")
    agg = sub.groupby([geo_col, time_col])[val_col].mean().reset_index()
    year, series = _pick_year(agg, geo_col, time_col, val_col)
    series.name = "artificial_land_pct"
    return year or "2018", series


def parse_doing_business() -> pd.Series:
    """D09: World Bank DB2020 — ease of doing business score, country-level."""
    df = pd.read_csv(RAW / "world_bank" / "doing_business_2020.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    s = df.set_index("country_code")["ease_of_doing_business_score"]
    s.name = "doing_business_score"
    return s


def parse_cit_rates() -> pd.Series:
    """D10: OECD CIT rates — headline statutory rate (%), country-level."""
    df = pd.read_csv(RAW / "oecd" / "corporate_tax_rates.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    # Detect CIT rate column
    rate_col = next((c for c in df.columns if "rate" in c or "cit" in c), df.columns[1])
    cc_col = next((c for c in df.columns if "country" in c or "code" in c), df.columns[0])
    s = df.set_index(cc_col)[rate_col]
    s.name = "cit_rate_pct"
    return s


def parse_ntc() -> pd.Series:
    """D03: ENTSO-E NTC — import capacity (MW), country-level."""
    df = pd.read_csv(RAW / "entso_e" / "ntc_capacity.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    cc_col = next((c for c in df.columns if "country" in c or "code" in c), df.columns[0])
    val_col = next((c for c in df.columns if "import" in c), df.columns[1])
    s = pd.to_numeric(df.set_index(cc_col)[val_col], errors="coerce")
    s.name = "ntc_import_mw"
    return s


def parse_wei2() -> pd.Series:
    """D04: EEA WEI2 — water exploitation index, country-level (lower = better)."""
    df = pd.read_csv(RAW / "manual" / "eea_wei2.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    cc_col = next((c for c in df.columns if "country" in c or "code" in c), df.columns[0])
    val_col = "wei2_value"
    s = pd.to_numeric(df.set_index(cc_col)[val_col], errors="coerce")
    s.name = "water_exploitation_index"
    return s


def parse_esif() -> pd.Series:
    """D08: EC ESIF 2014-2020 absorption rate, NUTS2-level where available."""
    df = pd.read_csv(RAW / "manual" / "esif_absorption.csv", low_memory=False)
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    # Find NUTS2 and financial columns
    nuts_col = next((c for c in df.columns if "nut" in c and "2" in c), None)
    if nuts_col is None:
        nuts_col = next((c for c in df.columns if "region" in c or "nuts" in c), None)
    plan_col = next((c for c in df.columns if "planned" in c or "allocation" in c), None)
    actual_col = next((c for c in df.columns if "payment" in c or "actual" in c or "expenditure" in c), None)
    if nuts_col is None or plan_col is None or actual_col is None:
        # Fallback: return empty series
        return pd.Series(name="esif_absorption_rate", dtype=float)
    df[plan_col] = pd.to_numeric(df[plan_col].astype(str).str.replace(",", "").str.replace(" ", ""), errors="coerce")
    df[actual_col] = pd.to_numeric(df[actual_col].astype(str).str.replace(",", "").str.replace(" ", ""), errors="coerce")
    # Keep 4-char codes that look like NUTS2
    nuts_mask = df[nuts_col].astype(str).str.match(r"^[A-Z]{2}[A-Z0-9]{2}$")
    sub = df[nuts_mask].copy()
    if sub.empty:
        # Use 2-char country codes as fallback
        sub = df.copy()
        sub["country_code"] = df[nuts_col].astype(str).str[:2]
        group = sub.groupby("country_code")[[plan_col, actual_col]].sum()
        rate = (group[actual_col] / group[plan_col].replace(0, np.nan)).clip(0, 1)
        rate.name = "esif_absorption_rate"
        return rate
    group = sub.groupby(nuts_col)[[plan_col, actual_col]].sum()
    rate = (group[actual_col] / group[plan_col].replace(0, np.nan)).clip(0, 1)
    rate.name = "esif_absorption_rate"
    return rate


def parse_nuclear_capacity() -> pd.Series:
    """D13: IAEA PRIS — aggregate operational nuclear capacity (MW) per NUTS2."""
    df = pd.read_csv(RAW / "manual" / "iaea_pris_plants.csv")
    operational = df[df["status"] == "operational"]
    agg = operational.groupby("nuts2_code")["net_capacity_mw"].sum()
    agg.name = "nuclear_capacity_mw"
    return agg


def parse_eurohpc_proximity(nuts2_codes: pd.Index) -> pd.Series:
    """D14: EuroHPC — score per NUTS2: 1.0 if site in NUTS2, 0.6 if country match, else 0."""
    sites = pd.read_csv(RAW / "manual" / "eurohpc_sites.csv")
    # Build country -> max tier map (world_class=1.0, pre_petascale=0.7, national=0.5)
    tier_score = {"world_class": 1.0, "pre_petascale": 0.7, "national": 0.5}
    country_max: dict[str, float] = {}
    nuts2_exact: dict[str, float] = {}
    for _, row in sites.iterrows():
        ts = tier_score.get(row["tier"], 0.4)
        cc = row["country_code"]
        n2 = row["nuts2_code"]
        country_max[cc] = max(country_max.get(cc, 0.0), ts * 0.6)
        nuts2_exact[n2] = max(nuts2_exact.get(n2, 0.0), ts)
    result = {}
    for code in nuts2_codes:
        if code in nuts2_exact:
            result[code] = nuts2_exact[code]
        else:
            cc = code[:2]
            result[code] = country_max.get(cc, 0.0)
    s = pd.Series(result, name="eurohpc_proximity_score")
    return s


def parse_quantum_flags(nuts2_country_map: pd.Series) -> pd.Series:
    """D15: Quantum programme tier (0-3) broadcast to NUTS2."""
    df = pd.read_csv(RAW / "manual" / "national_quantum_flags.csv")
    cc_to_tier = df.set_index("country_code")["tier"].to_dict()
    s = nuts2_country_map.map(cc_to_tier).fillna(0).astype(int)
    s.name = "quantum_programme_tier"
    return s


def parse_nuclear_policy(nuts2_country_map: pd.Series) -> dict[str, pd.Series]:
    """D16: Nuclear policy flags broadcast to NUTS2."""
    df = pd.read_csv(RAW / "manual" / "nuclear_policy_flags.csv")
    policy_map = df.set_index("country_code")["nuclear_policy"].to_dict()
    eligible_map = df.set_index("country_code")["haleu_smr_eligible"].to_dict()
    policy = nuts2_country_map.map(policy_map).fillna("NO_NUCLEAR")
    policy.name = "nuclear_policy"
    eligible = nuts2_country_map.map(eligible_map).fillna(False).astype(bool)
    eligible.name = "haleu_smr_eligible"
    return {"nuclear_policy": policy, "haleu_smr_eligible": eligible}


# ─── Country → NUTS2 broadcast helper ────────────────────────────────────────

def broadcast_country_to_nuts2(country_series: pd.Series,
                                nuts2_country_map: pd.Series) -> pd.Series:
    """Map a country-indexed Series to NUTS2-indexed Series."""
    return nuts2_country_map.map(country_series.to_dict())


# ─── k=4 Clustering ──────────────────────────────────────────────────────────

def _label_clusters(centroids: np.ndarray, k: int) -> list[str]:
    """
    Assign descriptive labels based on centroid profiles.
    Uses relative ranking of overall level (mean of 4 scores) and
    whether a cluster leads on cost vs. cluster/talent dimensions.
    """
    means = centroids.mean(axis=1)
    rank = np.argsort(means)[::-1]  # highest overall first

    level_labels = {
        0: "Frontier innovation regions",
        1: "Established industrial regions",
        2: "Emerging capacity regions",
        3: "Catching-up & peripheral regions",
    }
    labels = [""] * k
    for position, cluster_idx in enumerate(rank):
        labels[cluster_idx] = level_labels.get(position, f"Cluster {cluster_idx}")
    return labels


def run_clustering(silver: pd.DataFrame) -> pd.DataFrame:
    """Fit k=4 k-means on 4 upstream dimension scores. Returns cluster assignments."""
    X = silver[SCORE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=4, n_init=50, max_iter=500, random_state=42)
    km.fit(X_scaled)
    labels_k4 = km.labels_

    sil_mean = silhouette_score(X_scaled, labels_k4)
    sil_samples = silhouette_samples(X_scaled, labels_k4)

    # ARI vs original k=2 archetype
    ari_vs_k2 = adjusted_rand_score(silver["archetype_id"].values, labels_k4)

    # Centroid profiles in original (z-score) units
    centroids_scaled = km.cluster_centers_
    centroids_orig = scaler.inverse_transform(centroids_scaled)

    archetype_labels = _label_clusters(centroids_orig, 4)

    diag = []
    for i in range(4):
        mask = labels_k4 == i
        diag.append({
            "archetype4_id": i,
            "archetype4_label": archetype_labels[i],
            "n_regions": int(mask.sum()),
            "mean_silhouette": float(sil_samples[mask].mean()),
            **{f"centroid_{c}": float(centroids_orig[i, j]) for j, c in enumerate(SCORE_COLS)},
        })

    diag_df = pd.DataFrame(diag)
    diag_df["silhouette_overall"] = sil_mean
    diag_df["ari_vs_k2"] = ari_vs_k2
    diag_df.to_csv(ANALYSIS / "p2_clustering_diagnostics.csv", index=False)

    result = pd.DataFrame({
        "nuts2_code": silver.index,
        "archetype4_id": labels_k4,
        "archetype4_label": [archetype_labels[l] for l in labels_k4],
        "archetype4_silhouette": sil_samples,
        "archetype4_distance_to_centroid": np.linalg.norm(
            X_scaled - centroids_scaled[labels_k4], axis=1
        ),
    }).set_index("nuts2_code")

    return result, sil_mean, ari_vs_k2


# ─── Main pipeline ────────────────────────────────────────────────────────────

def main() -> None:
    ingestion_report: dict = {"features": {}, "decisions": []}
    year_alignment: list[dict] = []

    # ── Step 1: Load upstream Gold ────────────────────────────────────────────
    with pipeline_step("p2_load_upstream"):
        gold = pd.read_parquet(UPSTREAM_GOLD)
        gold.index.name = "nuts2_code"
        nuts2_country = gold["country_code"]  # nuts2_code → country_code

    # ── Step 2: Ingest new datasets ───────────────────────────────────────────
    new_features: dict[str, pd.Series] = {}

    with pipeline_step("p2_eurostat_features"):

        # Electricity price
        try:
            yr, s = parse_electricity_price()
            s = broadcast_country_to_nuts2(s, nuts2_country)
            new_features["electricity_price_eur_kwh"] = s
            year_alignment.append({"feature": "electricity_price_eur_kwh", "source": "nrg_pc_205", "year": yr, "nuts_level": "country"})
        except Exception as e:
            ingestion_report["decisions"].append({"feature": "electricity_price_eur_kwh", "action": "SKIP", "reason": str(e)})

        # Renewable share
        try:
            yr, s = parse_renewable_share()
            s = broadcast_country_to_nuts2(s, nuts2_country)
            new_features["renewable_energy_share_pct"] = s
            year_alignment.append({"feature": "renewable_energy_share_pct", "source": "nrg_ind_ren", "year": yr, "nuts_level": "country"})
        except Exception as e:
            ingestion_report["decisions"].append({"feature": "renewable_energy_share_pct", "action": "SKIP", "reason": str(e)})

        # Rail freight
        try:
            yr, s = parse_rail_freight()
            # NOTE (2026-09-26 fix): no longer .fillna(0.0) here -- a region
            # genuinely missing from tran_r_rago (no reported rail freight
            # data) is not the same as a region with real zero rail freight
            # traffic. Leaving it NaN lets the Step 5 generic imputation
            # (country median, then EU median) impute it properly and set
            # rail_freight_ktonnes_imputed_flag=True, instead of silently
            # asserting a false "real" zero for every unmatched region.
            s = s.reindex(gold.index)
            new_features["rail_freight_ktonnes"] = s
            year_alignment.append({"feature": "rail_freight_ktonnes", "source": "tran_r_rago", "year": yr, "nuts_level": "nuts2"})
        except Exception as e:
            ingestion_report["decisions"].append({"feature": "rail_freight_ktonnes", "action": "SKIP", "reason": str(e)})

        # Port throughput
        try:
            yr, s = parse_port_throughput()
            s = broadcast_country_to_nuts2(s, nuts2_country)
            new_features["port_throughput_ktonnes"] = s
            year_alignment.append({"feature": "port_throughput_ktonnes", "source": "mar_go_aa", "year": yr, "nuts_level": "country"})
        except Exception as e:
            ingestion_report["decisions"].append({"feature": "port_throughput_ktonnes", "action": "SKIP", "reason": str(e)})

        # GERD by sector
        try:
            yr, sectors = parse_gerd_by_sector()
            for name, s in sectors.items():
                s2 = broadcast_country_to_nuts2(s, nuts2_country)
                new_features[name] = s2
                year_alignment.append({"feature": name, "source": "rd_e_gerdsc", "year": yr, "nuts_level": "country"})
        except Exception as e:
            ingestion_report["decisions"].append({"feature": "gerd_sectors", "action": "SKIP", "reason": str(e)})

        # Ultra-fast broadband
        try:
            yr, s = parse_ultrafast_broadband()
            s = s.reindex(gold.index)
            new_features["ultrafast_broadband_pct"] = s
            year_alignment.append({"feature": "ultrafast_broadband_pct", "source": "isoc_r_broad_h", "year": yr, "nuts_level": "nuts2"})
        except Exception as e:
            ingestion_report["decisions"].append({"feature": "ultrafast_broadband_pct", "action": "SKIP", "reason": str(e)})

        # Land cover
        try:
            yr, s = parse_land_cover()
            s = s.reindex(gold.index)
            new_features["artificial_land_pct"] = s
            year_alignment.append({"feature": "artificial_land_pct", "source": "lan_lcv_ovw", "year": yr, "nuts_level": "nuts2"})
        except Exception as e:
            ingestion_report["decisions"].append({"feature": "artificial_land_pct", "action": "SKIP", "reason": str(e)})

    with pipeline_step("p2_non_eurostat_features"):

        for name, fn in [("doing_business_score", parse_doing_business),
                          ("cit_rate_pct", parse_cit_rates),
                          ("ntc_import_mw", parse_ntc),
                          ("water_exploitation_index", parse_wei2)]:
            try:
                s = fn()
                s2 = broadcast_country_to_nuts2(s, nuts2_country)
                new_features[name] = s2
                year_alignment.append({"feature": name, "source": name, "year": "2020-2022", "nuts_level": "country"})
            except Exception as e:
                ingestion_report["decisions"].append({"feature": name, "action": "SKIP", "reason": str(e)})

        # ESIF absorption rate
        try:
            s = parse_esif()
            if len(s) > 0:
                if s.index.str.len().max() == 4:
                    s = s.reindex(gold.index).fillna(np.nan)
                else:
                    s = broadcast_country_to_nuts2(s, nuts2_country)
                new_features["esif_absorption_rate"] = s
                year_alignment.append({"feature": "esif_absorption_rate", "source": "esif_2014_2020", "year": "2022", "nuts_level": "nuts2_approx"})
        except Exception as e:
            ingestion_report["decisions"].append({"feature": "esif_absorption_rate", "action": "SKIP", "reason": str(e)})

        # Nuclear capacity
        try:
            s = parse_nuclear_capacity().reindex(gold.index).fillna(0.0)
            new_features["nuclear_capacity_mw"] = s
            year_alignment.append({"feature": "nuclear_capacity_mw", "source": "iaea_pris", "year": "2024", "nuts_level": "nuts2"})
        except Exception as e:
            ingestion_report["decisions"].append({"feature": "nuclear_capacity_mw", "action": "SKIP", "reason": str(e)})

        # EuroHPC proximity
        try:
            s = parse_eurohpc_proximity(gold.index)
            new_features["eurohpc_proximity_score"] = s
            year_alignment.append({"feature": "eurohpc_proximity_score", "source": "eurohpc_sites", "year": "2024", "nuts_level": "nuts2"})
        except Exception as e:
            ingestion_report["decisions"].append({"feature": "eurohpc_proximity_score", "action": "SKIP", "reason": str(e)})

        # Quantum flags
        try:
            s = parse_quantum_flags(nuts2_country)
            new_features["quantum_programme_tier"] = s
            year_alignment.append({"feature": "quantum_programme_tier", "source": "quantum_flags", "year": "2024", "nuts_level": "country"})
        except Exception as e:
            ingestion_report["decisions"].append({"feature": "quantum_programme_tier", "action": "SKIP", "reason": str(e)})

        # Nuclear policy flags
        try:
            policy_dict = parse_nuclear_policy(nuts2_country)
            for name, s in policy_dict.items():
                new_features[name] = s
                year_alignment.append({"feature": name, "source": "nuclear_policy", "year": "2024", "nuts_level": "country"})
        except Exception as e:
            ingestion_report["decisions"].append({"feature": "nuclear_policy", "action": "SKIP", "reason": str(e)})

    # ── Step 3: Build Silver ──────────────────────────────────────────────────
    with pipeline_step("p2_build_silver"):
        silver = gold.copy()
        for name, series in new_features.items():
            silver[name] = series.reindex(silver.index)

        null_rates = (silver.isnull().sum() / len(silver)).round(4).to_dict()
        for f, nr in null_rates.items():
            ingestion_report["features"][f] = {"null_rate": nr}

        SILVER_PATH.parent.mkdir(parents=True, exist_ok=True)
        silver.to_parquet(SILVER_PATH)

    # ── Step 4: k=4 Clustering ────────────────────────────────────────────────
    with pipeline_step("p2_clustering_k4"):
        cluster_df, sil_mean, ari_vs_k2 = run_clustering(silver)
        silver_clustered = silver.join(cluster_df)

    # ── Step 5: Build Gold ────────────────────────────────────────────────────
    with pipeline_step("p2_build_gold"):
        # Imputation flags for new numeric features
        for name in new_features:
            if silver_clustered[name].dtype in [np.float64, np.float32, float]:
                flag_col = f"{name}_imputed_flag"
                silver_clustered[flag_col] = silver_clustered[name].isnull()
                # Impute missing with country median
                cc = silver_clustered["country_code"]
                country_median = silver_clustered.groupby(cc)[name].transform("median")
                silver_clustered[name] = silver_clustered[name].fillna(country_median)
                # Remaining nulls (whole-country gap): fill with EU median
                eu_median = silver_clustered[name].median()
                silver_clustered[name] = silver_clustered[name].fillna(eu_median)

        GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
        silver_clustered.to_parquet(GOLD_PATH)

    # ── Step 6: Write artifacts ───────────────────────────────────────────────
    pd.DataFrame(year_alignment).to_csv(ANALYSIS / "p2_year_alignment.csv", index=False)
    (ANALYSIS / "p2_ingestion_report.json").write_text(
        json.dumps(ingestion_report, indent=2, default=str), encoding="utf-8"
    )

    # ── Gate check ────────────────────────────────────────────────────────────
    gold_out = pd.read_parquet(GOLD_PATH)
    n_rows, n_cols = gold_out.shape
    n_new = len(new_features)
    null_total = gold_out.isnull().sum().sum()
    null_pct = round(100 * null_total / (n_rows * n_cols), 2)

    diag = pd.read_csv(ANALYSIS / "p2_clustering_diagnostics.csv")

    print()
    print("=" * 65)
    print("P2 INGESTION & CLUSTERING GATE CHECK")
    print("=" * 65)
    print(f"  Silver shape          : {silver.shape}")
    print(f"  Gold shape            : {gold_out.shape}")
    print(f"  New features ingested : {n_new}")
    print(f"  Skipped features      : {len(ingestion_report['decisions'])}")
    print(f"  Null rate (Gold)      : {null_pct}%")
    print()
    print("  k=4 Clustering (on score_cost, score_talent, score_infra, score_cluster):")
    print(f"    Silhouette (mean)   : {sil_mean:.4f}")
    print(f"    ARI vs k=2          : {ari_vs_k2:.4f}")
    print(f"    NOTE: k=2 silhouette was 0.409 — k=4 interpretability override documented")
    print()
    print("  Archetype profiles:")
    for _, row in diag.iterrows():
        print(f"    A{int(row['archetype4_id'])}: {row['archetype4_label']:<40} "
              f"n={int(row['n_regions']):3d}  sil={row['mean_silhouette']:.3f}")
    print()

    if ingestion_report["decisions"]:
        print("  Skipped (check ingestion_report.json):")
        for d in ingestion_report["decisions"][:5]:
            print(f"    {d['feature']}: {d['reason'][:60]}")

    gate_pass = (n_rows == 242 and n_new >= 12 and null_pct < 15.0)
    print()
    print(f"GATE_P2={'PASS' if gate_pass else 'FAIL'}")
    if not gate_pass:
        if n_rows != 242:
            print(f"  REASON: expected 242 rows, got {n_rows}")
        if n_new < 12:
            print(f"  REASON: expected >=12 new features, got {n_new}")
        if null_pct >= 15.0:
            print(f"  REASON: null rate {null_pct}% >= 15% threshold")
    print("=" * 65)

    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
