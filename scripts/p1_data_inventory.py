"""
P1 — Data Inventory & Source Catalog.

Loads the EU-Innovation-Panel Gold layer, enumerates all 66 upstream features,
maps them to the 8 disruptor types, catalogs 12 new datasets (to_download),
computes feature-type coverage, and writes the master data catalog.

Outputs:
  analysis/p1_data_catalog.csv        — master catalog with 78+ entries
  analysis/p1_type_coverage_matrix.csv — feature presence per type (8 cols)
  analysis/p1_data_gaps_register.csv  — DATA GAP register with proxies
  analysis/p1_upstream_feature_map.csv — Gold layer feature × type mapping
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logging_config import pipeline_step  # noqa: E402

UPSTREAM_GOLD = ROOT.parent / "EU-Innovation-Panel" / "data" / "gold" / "region_profiles_gold.parquet"
ANALYSIS_DIR = ROOT / "analysis"

# ─── Type membership per feature (True = contributes to that type) ───────────
# Columns: T1=AI, T2=Bio, T3=Semi, T4=Clean, T5=DC, T6=HALEU, T7=DeepTech, T8=Quantum
UPSTREAM_FEATURE_TYPE_MAP: dict[str, dict[str, bool]] = {
    # Prosperity / Cost dimension
    "gdp_per_capita_pps":         dict(T1=True,  T2=True,  T3=True,  T4=True,  T5=True,  T6=True,  T7=True,  T8=True),
    "score_prosperity":           dict(T1=True,  T2=True,  T3=True,  T4=True,  T5=True,  T6=True,  T7=True,  T8=True),
    # Talent dimension
    "hrst_per_1000":              dict(T1=True,  T2=True,  T3=True,  T4=False, T5=False, T6=False, T7=True,  T8=True),
    "tertiary_enrolment_rate":    dict(T1=True,  T2=True,  T3=False, T4=False, T5=False, T6=False, T7=True,  T8=True),
    "employment_rate":            dict(T1=True,  T2=True,  T3=True,  T4=True,  T5=True,  T6=True,  T7=True,  T8=True),
    "score_talent":               dict(T1=True,  T2=True,  T3=True,  T4=False, T5=False, T6=False, T7=True,  T8=True),
    # Digital Infrastructure
    "broadband_penetration_pct":  dict(T1=True,  T2=False, T3=False, T4=False, T5=True,  T6=False, T7=False, T8=True),
    "enterprise_internet_use":    dict(T1=True,  T2=False, T3=False, T4=False, T5=True,  T6=False, T7=False, T8=False),
    "gva_ict_share":              dict(T1=True,  T2=False, T3=False, T4=False, T5=True,  T6=False, T7=False, T8=False),
    "score_digital_infra":        dict(T1=True,  T2=False, T3=False, T4=False, T5=True,  T6=False, T7=False, T8=True),
    # R&D / Innovation cluster
    "rd_expenditure_pct_gdp":     dict(T1=True,  T2=True,  T3=True,  T4=True,  T5=False, T6=True,  T7=True,  T8=True),
    "business_rd_pct_gdp":        dict(T1=True,  T2=True,  T3=True,  T4=True,  T5=False, T6=True,  T7=True,  T8=True),
    "epo_patents_per_mio_pop":    dict(T1=True,  T2=True,  T3=True,  T4=True,  T5=False, T6=False, T7=True,  T8=True),
    "score_innovation_cluster":   dict(T1=True,  T2=True,  T3=True,  T4=True,  T5=False, T6=True,  T7=True,  T8=True),
    # Sector LQs
    "lq_nace_j62j63":             dict(T1=True,  T2=False, T3=False, T4=False, T5=True,  T6=False, T7=False, T8=False),
    "lq_nace_c21_m72":            dict(T1=False, T2=True,  T3=False, T4=False, T5=False, T6=False, T7=False, T8=False),
    "lq_nace_c26":                dict(T1=False, T2=False, T3=True,  T4=False, T5=False, T6=False, T7=True,  T8=False),
    "lq_nace_d35_clean":          dict(T1=False, T2=False, T3=False, T4=True,  T5=True,  T6=True,  T7=False, T8=False),
    # Population / Density
    "population":                 dict(T1=False, T2=False, T3=False, T4=False, T5=False, T6=False, T7=False, T8=False),
    "population_density":         dict(T1=False, T2=False, T3=True,  T4=False, T5=True,  T6=True,  T7=False, T8=False),
    # Hi-tech employment
    "hi_tech_employment_pct":     dict(T1=True,  T2=True,  T3=True,  T4=False, T5=False, T6=False, T7=True,  T8=True),
    # Archetype from base study (contextual feature, not direct suitability input)
    "archetype_id":               dict(T1=False, T2=False, T3=False, T4=False, T5=False, T6=False, T7=False, T8=False),
    "archetype_label":            dict(T1=False, T2=False, T3=False, T4=False, T5=False, T6=False, T7=False, T8=False),
    "is_transition_region":       dict(T1=False, T2=False, T3=False, T4=True,  T5=False, T6=False, T7=False, T8=False),
    "silhouette_sample":          dict(T1=False, T2=False, T3=False, T4=False, T5=False, T6=False, T7=False, T8=False),
    "distance_to_centroid":       dict(T1=False, T2=False, T3=False, T4=False, T5=False, T6=False, T7=False, T8=False),
    "country_code":               dict(T1=False, T2=False, T3=False, T4=False, T5=False, T6=False, T7=False, T8=False),
    "nuts2_name":                 dict(T1=False, T2=False, T3=False, T4=False, T5=False, T6=False, T7=False, T8=False),
}

# ─── New datasets to download ─────────────────────────────────────────────────
NEW_DATASETS: list[dict] = [
    {
        "id": "D01", "group": "energy",
        "label": "Electricity prices — industrial (incl. taxes), bi-annual",
        "source": "Eurostat", "code": "nrg_pc_205",
        "reference_year": "2022", "nuts_level": "country",
        "types": ["T3", "T4", "T5", "T6"],
        "status": "to_download",
        "local_path": "data/raw/eurostat/nrg_pc_205.csv",
        "license": "Eurostat — free reuse with attribution",
        "access_method": "bulk_sdmx_csv",
        "null_rate_estimate": 0.05,
        "notes": "Band ID=IE (medium industrial consumers). Country-level; assign to NUTS2 via country_code.",
    },
    {
        "id": "D02", "group": "energy",
        "label": "Renewable electricity share by country",
        "source": "Eurostat", "code": "nrg_ind_ren",
        "reference_year": "2022", "nuts_level": "country",
        "types": ["T4", "T5", "T6"],
        "status": "to_download",
        "local_path": "data/raw/eurostat/nrg_ind_ren.csv",
        "license": "Eurostat — free reuse with attribution",
        "access_method": "bulk_sdmx_csv",
        "null_rate_estimate": 0.05,
        "notes": "Renewable share of gross final energy consumption. Country-level proxy for NUTS2.",
    },
    {
        "id": "D03", "group": "energy",
        "label": "ENTSO-E Cross-Border NTC capacity by interconnection",
        "source": "ENTSO-E Transparency Platform", "code": "Physical_Flows_NTC",
        "reference_year": "2022", "nuts_level": "country",
        "types": ["T5", "T6"],
        "status": "to_download",
        "local_path": "data/raw/entso_e/ntc_capacity.csv",
        "license": "ENTSO-E — open data, CC BY 4.0",
        "access_method": "manual_bulk_csv",
        "null_rate_estimate": 0.10,
        "notes": "Download from transparency.entsoe.eu > Cross-border flows > Planned Exchange. "
                 "Country-level only; aggregate by importing country for grid surplus proxy.",
    },
    {
        "id": "D04", "group": "water",
        "label": "Water Exploitation Index (WEI2) by river basin",
        "source": "EEA", "code": "WEI2",
        "reference_year": "2019", "nuts_level": "country",
        "types": ["T5"],
        "status": "to_download",
        "local_path": "data/raw/manual/eea_wei2.csv",
        "license": "EEA — open data, CC BY 4.0",
        "access_method": "manual_download",
        "null_rate_estimate": 0.20,
        "notes": "WEI2 < 0.2 = no stress; >0.4 = severe stress. Country-level or basin-level. "
                 "Inverse-score for cooling water availability.",
    },
    {
        "id": "D05", "group": "transport",
        "label": "Rail freight transport by NUTS2 region",
        "source": "Eurostat", "code": "tran_r_rapa",
        "reference_year": "2021", "nuts_level": "nuts2",
        "types": ["T3", "T7"],
        "status": "to_download",
        "local_path": "data/raw/eurostat/tran_r_rapa.csv",
        "license": "Eurostat — free reuse with attribution",
        "access_method": "bulk_sdmx_csv",
        "null_rate_estimate": 0.25,
        "notes": "Tonnes of goods loaded/unloaded by NUTS2. High sparsity expected for landlocked regions.",
    },
    {
        "id": "D06", "group": "transport",
        "label": "Sea port goods throughput by port",
        "source": "Eurostat", "code": "mar_go_aa",
        "reference_year": "2022", "nuts_level": "port",
        "types": ["T3"],
        "status": "to_download",
        "local_path": "data/raw/eurostat/mar_go_aa.csv",
        "license": "Eurostat — free reuse with attribution",
        "access_method": "bulk_sdmx_csv",
        "null_rate_estimate": 0.30,
        "notes": "Port-level data. Geocode port to NUTS2 via GISCO. "
                 "For landlocked NUTS2: assign nearest-port distance as inverse proxy.",
    },
    {
        "id": "D07", "group": "rd",
        "label": "GERD by sector of performance (incl. DEF)",
        "source": "Eurostat", "code": "rd_e_gerdsc",
        "reference_year": "2021", "nuts_level": "country",
        "types": ["T2", "T6", "T7", "T8"],
        "status": "to_download",
        "local_path": "data/raw/eurostat/rd_e_gerdsc.csv",
        "license": "Eurostat — free reuse with attribution",
        "access_method": "bulk_sdmx_csv",
        "null_rate_estimate": 0.15,
        "notes": "Sector = GOV + HES (T2, T7, T8). Sector = DEF (T6 — defence R&D proxy). "
                 "Country-level only; assign to NUTS2 via country_code.",
    },
    {
        "id": "D08", "group": "policy",
        "label": "ESIF 2014-2020 structural funds absorption rate by NUTS2",
        "source": "EC Open Data / Cohesion Data", "code": "ESIF_2014_2020",
        "reference_year": "2022", "nuts_level": "nuts2",
        "types": ["T4"],
        "status": "to_download",
        "local_path": "data/raw/manual/esif_absorption.csv",
        "license": "EC — open data, CC BY 4.0",
        "access_method": "manual_bulk_csv",
        "null_rate_estimate": 0.15,
        "notes": "Download from cohesiondata.ec.europa.eu. "
                 "Absorption rate = actual payments / planned allocation. Proxy for policy execution capacity.",
    },
    {
        "id": "D09", "group": "regulatory",
        "label": "Ease of Doing Business index by country (DB2020)",
        "source": "World Bank", "code": "DB2020",
        "reference_year": "2020", "nuts_level": "country",
        "types": ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"],
        "status": "to_download",
        "local_path": "data/raw/world_bank/doing_business_2020.csv",
        "license": "World Bank — open data, CC BY 4.0",
        "access_method": "bulk_csv",
        "null_rate_estimate": 0.05,
        "notes": "Last published edition of Doing Business. Country-level; assign to NUTS2 via country_code.",
    },
    {
        "id": "D10", "group": "fiscal",
        "label": "Corporate income tax rate (headline + OECD effective)",
        "source": "OECD Tax Database", "code": "Table_II_1",
        "reference_year": "2022", "nuts_level": "country",
        "types": ["T1", "T2", "T3", "T5"],
        "status": "to_download",
        "local_path": "data/raw/oecd/corporate_tax_rates.csv",
        "license": "OECD — free reuse with attribution",
        "access_method": "bulk_csv",
        "null_rate_estimate": 0.05,
        "notes": "Headline statutory CIT rate. Country-level. Inverse-score for fiscal attractiveness.",
    },
    {
        "id": "D11", "group": "digital",
        "label": "Broadband ultra-fast coverage (>100 Mbps) by NUTS2",
        "source": "Eurostat", "code": "isoc_r_broad_h",
        "reference_year": "2022", "nuts_level": "nuts2",
        "types": ["T1", "T5", "T8"],
        "status": "to_download",
        "local_path": "data/raw/eurostat/isoc_r_broad_h.csv",
        "license": "Eurostat — free reuse with attribution",
        "access_method": "bulk_sdmx_csv",
        "null_rate_estimate": 0.10,
        "notes": "% households covered by ultra-fast broadband (>100 Mbps NGA). NUTS2-level.",
    },
    {
        "id": "D12", "group": "land",
        "label": "Industrial land share (Corine Land Cover 2018 aggregated to NUTS2)",
        "source": "Copernicus / EEA", "code": "CLC2018",
        "reference_year": "2018", "nuts_level": "nuts2",
        "types": ["T3", "T5", "T6"],
        "status": "to_download",
        "local_path": "data/raw/manual/clc2018_nuts2_agg.csv",
        "license": "Copernicus — free reuse, CC BY 4.0",
        "access_method": "manual_spatial_join",
        "null_rate_estimate": 0.10,
        "notes": "CLC classes 121 (Industrial/commercial), 131 (Mineral extraction), 132 (Dump sites). "
                 "Requires spatial join of CLC raster to NUTS2 polygons. "
                 "Download CLC2018 GeoTIFF from land.copernicus.eu.",
    },
    {
        "id": "D13", "group": "nuclear",
        "label": "Nuclear power plant locations and capacity (IAEA PRIS)",
        "source": "IAEA PRIS", "code": "PRIS_reactors",
        "reference_year": "2024", "nuts_level": "point",
        "types": ["T6"],
        "status": "to_download",
        "local_path": "data/raw/manual/iaea_pris_plants.csv",
        "license": "IAEA — public access, non-commercial",
        "access_method": "manual_download",
        "null_rate_estimate": 0.0,
        "notes": "Download reactor list from pris.iaea.org/PRIS/WorldStatistics. "
                 "Geocode reactor locations to NUTS2 via GISCO. "
                 "Compute: min_distance_nuclear_mw (weighted by capacity) per NUTS2.",
    },
    {
        "id": "D14", "group": "compute",
        "label": "EuroHPC Joint Undertaking supercomputing sites",
        "source": "EuroHPC JU", "code": "EuroHPC_sites",
        "reference_year": "2024", "nuts_level": "point",
        "types": ["T8"],
        "status": "manual_compile",
        "local_path": "data/raw/manual/eurohpc_sites.csv",
        "license": "EuroHPC JU — public information",
        "access_method": "manual_list",
        "null_rate_estimate": 0.0,
        "notes": "~30 EuroHPC petascale and pre-petascale sites across EU. "
                 "List available at eurohpc-ju.europa.eu/supercomputers. "
                 "Geocode to NUTS2; compute proximity score (inverse distance decay).",
    },
    {
        "id": "D15", "group": "quantum",
        "label": "National quantum technology programme flag",
        "source": "Manual compilation from EC Quantum Flagship", "code": "quantum_flag",
        "reference_year": "2024", "nuts_level": "country",
        "types": ["T8"],
        "status": "manual_compile",
        "local_path": "data/raw/manual/national_quantum_flags.csv",
        "license": "Public domain — compiled from official programme websites",
        "access_method": "manual_binary_flag",
        "null_rate_estimate": 0.0,
        "notes": "Binary flag: country has dedicated national quantum programme (1) or not (0). "
                 "Sources: EC Quantum Flagship page, national NQI websites. "
                 "Countries with flagship programmes as of 2024: DE, FR, NL, FI, SE, DK, AT, ES, IT, PL.",
    },
    {
        "id": "D16", "group": "nuclear",
        "label": "Nuclear energy regulatory policy by country",
        "source": "Manual compilation from national energy laws", "code": "nuclear_policy_flag",
        "reference_year": "2024", "nuts_level": "country",
        "types": ["T6"],
        "status": "manual_compile",
        "local_path": "data/raw/manual/nuclear_policy_flags.csv",
        "license": "Public domain — compiled from official government sources",
        "access_method": "manual_categorical",
        "null_rate_estimate": 0.0,
        "notes": "Categories: PRO (actively expanding nuclear), NEUTRAL (no phase-out), "
                 "PHASE_OUT (Germany phase-out completed 2023, Belgium planning), "
                 "BANNED (Austria — constitutional ban, Italy). "
                 "HALEU/SMR suitability requires policy != BANNED.",
    },
]

# ─── DATA GAP register ────────────────────────────────────────────────────────
DATA_GAPS: list[dict] = [
    {
        "variable": "quantum_patents_nuts2_ipc",
        "type_affected": "T8",
        "gap_severity": 3,
        "reason": "No public NUTS2 IPC sub-class patent data (G06N10, H01L, B82Y) from free sources",
        "proxy": "EPO total patent density (epo_patents_per_mio_pop) + EuroHPC proximity score + national_quantum_flag",
        "proxy_quality": "moderate",
        "uncertainty_label": "high",
    },
    {
        "variable": "robotics_patents_nuts2_ipc",
        "type_affected": "T7",
        "gap_severity": 2,
        "reason": "No public NUTS2 IPC sub-class patent data (B25J) from free sources",
        "proxy": "machinery/automotive LQ (lq_nace_c26 partial) + BERD + CORDIS EIC funding (if available)",
        "proxy_quality": "low-moderate",
        "uncertainty_label": "high",
    },
    {
        "variable": "haleu_patents_nuts2_ipc",
        "type_affected": "T6",
        "gap_severity": 2,
        "reason": "No NUTS2 IPC nuclear patent sub-class from free sources",
        "proxy": "nuclear_capacity_proximity + chemical_lq (C20) + nuclear_policy_flag",
        "proxy_quality": "low-moderate",
        "uncertainty_label": "high",
    },
    {
        "variable": "grid_headroom_nuts2_mw",
        "type_affected": "T5",
        "gap_severity": 3,
        "reason": "No public NUTS2 available grid capacity dataset exists",
        "proxy": "ENTSO-E country NTC import surplus (inverse of net import balance) + electricity price index",
        "proxy_quality": "low",
        "uncertainty_label": "high",
    },
    {
        "variable": "industrial_land_cost_nuts2",
        "type_affected": "T3, T5, T6",
        "gap_severity": 2,
        "reason": "MSCI/CBRE industrial RE indices are licensed; no NUTS2 public benchmark",
        "proxy": "inverse(GDP per capita PPS) + CLC2018 industrial land share (D12)",
        "proxy_quality": "moderate",
        "uncertainty_label": "moderate",
    },
    {
        "variable": "vc_dealflow_nuts2",
        "type_affected": "T1, T2, T7",
        "gap_severity": 2,
        "reason": "Dealroom/Crunchbase are licensed; no free NUTS2 VC deal flow data",
        "proxy": "CORDIS EIC funding per NUTS2 (if CORDIS available) or business_rd_pct_gdp",
        "proxy_quality": "low-moderate",
        "uncertainty_label": "moderate",
    },
    {
        "variable": "cordis_horizon_eu_nuts2",
        "type_affected": "T1, T2, T7, T8",
        "gap_severity": 1,
        "reason": "CORDIS bulk CSV was 404 in EU-Innovation-Panel pipeline; URL relocated",
        "proxy": "CORDIS is publicly available at data.europa.eu/data/datasets — re-attempt download",
        "proxy_quality": "potentially full coverage if re-downloaded",
        "uncertainty_label": "low-moderate",
    },
]


def load_upstream_gold() -> pd.DataFrame:
    gold = pd.read_parquet(UPSTREAM_GOLD)
    return gold.reset_index()


def build_upstream_feature_map(gold_cols: list[str]) -> pd.DataFrame:
    records = []
    for col in gold_cols:
        row = {"feature": col, "source": "EU-Innovation-Panel Gold"}
        type_map = UPSTREAM_FEATURE_TYPE_MAP.get(col, {})
        for t in [f"T{i}" for i in range(1, 9)]:
            row[t] = type_map.get(t, False)
        row["any_type"] = any(type_map.get(t, False) for t in [f"T{i}" for i in range(1, 9)])
        records.append(row)
    return pd.DataFrame(records)


def build_master_catalog(gold_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in gold_cols:
        tm = UPSTREAM_FEATURE_TYPE_MAP.get(col, {})
        rows.append({
            "source_id": "upstream_gold",
            "variable_name": col,
            "feature_group": "upstream",
            "source_name": "EU-Innovation-Panel Gold Layer",
            "nuts_level": 2,
            "reference_year": "2019-2022",
            "access_method": "parquet_upstream",
            "license": "Eurostat / OECD — see EU-Innovation-Panel data catalog",
            "null_rate_estimate": 0.0,
            "update_frequency": "annual",
            "is_data_gap": False,
            "gap_severity": None,
            "types_served": ",".join(t for t in [f"T{i}" for i in range(1, 9)] if tm.get(t, False)),
            "status": "available",
        })

    for d in NEW_DATASETS:
        rows.append({
            "source_id": d["id"],
            "variable_name": d["code"],
            "feature_group": d["group"],
            "source_name": d["label"],
            "nuts_level": 2 if d["nuts_level"] == "nuts2" else "country/point",
            "reference_year": d["reference_year"],
            "access_method": d["access_method"],
            "license": d["license"],
            "null_rate_estimate": d["null_rate_estimate"],
            "update_frequency": "annual",
            "is_data_gap": False,
            "gap_severity": None,
            "types_served": ",".join(d["types"]),
            "status": d["status"],
        })

    for g in DATA_GAPS:
        rows.append({
            "source_id": f"GAP_{g['variable']}",
            "variable_name": g["variable"],
            "feature_group": "data_gap",
            "source_name": f"DATA GAP — proxy: {g['proxy'][:60]}...",
            "nuts_level": 2,
            "reference_year": "N/A",
            "access_method": "proxy",
            "license": "N/A",
            "null_rate_estimate": None,
            "update_frequency": "unknown",
            "is_data_gap": True,
            "gap_severity": g["gap_severity"],
            "types_served": g["type_affected"],
            "status": "data_gap",
        })

    return pd.DataFrame(rows)


def build_type_coverage_matrix(feature_map: pd.DataFrame) -> pd.DataFrame:
    type_cols = [f"T{i}" for i in range(1, 9)]
    coverage = {}
    for t in type_cols:
        contributing = feature_map[feature_map[t] == True]["feature"].tolist()  # noqa: E712
        coverage[t] = {
            "type_id": t,
            "n_features_upstream": len(contributing),
            "features": ", ".join(contributing[:10]) + ("..." if len(contributing) > 10 else ""),
            "n_new_datasets": sum(1 for d in NEW_DATASETS if t in d["types"]),
            "has_data_gap": any(t in g["type_affected"] for g in DATA_GAPS),
            "data_gap_severity_max": max(
                (g["gap_severity"] for g in DATA_GAPS if t in g["type_affected"]),
                default=0,
            ),
        }
    return pd.DataFrame(list(coverage.values())).set_index("type_id")


def main() -> None:
    with pipeline_step("p1_load_upstream"):
        gold = load_upstream_gold()
        gold_cols = [c for c in gold.columns if c != "nuts2_code"]

    with pipeline_step("p1_feature_map"):
        feature_map = build_upstream_feature_map(gold_cols)
        out_fm = ANALYSIS_DIR / "p1_upstream_feature_map.csv"
        feature_map.to_csv(out_fm, index=False)

    with pipeline_step("p1_catalog"):
        catalog = build_master_catalog(gold_cols)
        out_cat = ANALYSIS_DIR / "p1_data_catalog.csv"
        catalog.to_csv(out_cat, index=False)

    with pipeline_step("p1_type_coverage"):
        coverage = build_type_coverage_matrix(feature_map)
        out_cov = ANALYSIS_DIR / "p1_type_coverage_matrix.csv"
        coverage.to_csv(out_cov)

    with pipeline_step("p1_gap_register"):
        gap_df = pd.DataFrame(DATA_GAPS)
        out_gap = ANALYSIS_DIR / "p1_data_gaps_register.csv"
        gap_df.to_csv(out_gap, index=False)

    n_available = (catalog["status"] == "available").sum()
    n_to_download = catalog["status"].isin(["to_download", "manual_compile", "manual_spatial_join"]).sum()
    n_gap = catalog["is_data_gap"].sum()
    n_total = len(catalog)
    gap_pct = round(100 * n_gap / n_total, 1)

    print()
    print("=" * 65)
    print("P1 DATA INVENTORY GATE CHECK")
    print("=" * 65)
    print(f"  Total catalog entries : {n_total}")
    print(f"  Available (upstream)  : {n_available}")
    print(f"  To download / compile : {n_to_download}")
    print(f"  DATA GAPs             : {n_gap} ({gap_pct}% of catalog)")
    print()
    print("  Type coverage summary:")
    for _, row in coverage.iterrows():
        gap_flag = " [DATA GAP]" if row["has_data_gap"] else ""
        print(f"    {row.name}: {row['n_features_upstream']:2d} upstream features, "
              f"{row['n_new_datasets']} new datasets{gap_flag}")
    print()
    print(f"  Artifacts:")
    print(f"    {out_cat.relative_to(ROOT)}")
    print(f"    {out_fm.relative_to(ROOT)}")
    print(f"    {out_cov.relative_to(ROOT)}")
    print(f"    {out_gap.relative_to(ROOT)}")
    print()

    gate_pass = (n_total >= 40) and (gap_pct <= 25.0)
    print(f"GATE_P1={'PASS' if gate_pass else 'FAIL'}")
    if not gate_pass:
        print(f"  REASON: {'< 40 catalog entries' if n_total < 40 else f'DATA GAP rate {gap_pct}% > 25% threshold'}")
    print("=" * 65)

    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
