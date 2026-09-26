"""
P0 — Scaffold validation.
Verifies directory structure, upstream Gold layer availability, src/utils imports,
and writes the data catalog stub.
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

REQUIRED_DIRS = [
    "data/bronze",
    "data/raw/eurostat",
    "data/raw/oecd",
    "data/raw/entso_e",
    "data/raw/world_bank",
    "data/raw/manual",
    "data/silver",
    "data/gold",
    "scripts",
    "notebooks",
    "figures",
    "analysis",
    "reports",
    "references",
    "src/utils",
]

REQUIRED_FILES = [
    "CLAUDE.md",
    "Dockerfile",
    "Makefile",
    "environment.yml",
    "src/utils/logging_config.py",
    "src/utils/seed_check.py",
    "src/utils/vocab_guard.py",
]

# Eight disruptor ecosystem types (replaces 5-site-class taxonomy)
DISRUPTOR_TYPES = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
DISRUPTOR_TYPE_LABELS = {
    "T1": "AI / Machine Learning Hub",
    "T2": "Biotechnology / Life Sciences",
    "T3": "Semiconductors / Advanced Electronics",
    "T4": "Cleantech / Green Technology",
    "T5": "Hyperscale Data Centre Hub",
    "T6": "HALEU / Advanced Nuclear Industrial Base",
    "T7": "Deep-Tech Robotics Campus",
    "T8": "Quantum / Photonics Research Anchor",
}


def check_structure() -> list[str]:
    failures = []
    for d in REQUIRED_DIRS:
        p = ROOT / d
        if not p.is_dir():
            failures.append(f"MISSING DIR:  {d}")
    for f in REQUIRED_FILES:
        p = ROOT / f
        if not p.exists():
            failures.append(f"MISSING FILE: {f}")
    return failures


def check_upstream_gold() -> dict:
    if not UPSTREAM_GOLD.exists():
        return {"available": False, "path": str(UPSTREAM_GOLD)}
    gold = pd.read_parquet(UPSTREAM_GOLD)
    return {
        "available": True,
        "path": str(UPSTREAM_GOLD),
        "shape": list(gold.shape),
        "columns": list(gold.columns[:10]) + ["..."],
        "index_name": gold.index.name,
    }


def write_catalog_stub() -> None:
    catalog = {
        "project": "EU-MegaCampus-Siting",
        "created": "2026-05-24",
        "clustering_override": {
            "k": 4,
            "reason": "Four archetypes required for 8-type disruptor taxonomy mapping; k=2 conflates structurally distinct profiles on extended feature set.",
            "base_study_k": 2,
            "base_study_silhouette": 0.409,
        },
        "upstream_source": {
            "label": "EU-Innovation-Panel Gold Layer",
            "path": str(UPSTREAM_GOLD),
            "features": 66,
            "regions": 242,
            "status": "available",
        },
        "new_datasets": [
            {
                "id": "D01",
                "label": "Electricity prices — industrial (incl. taxes)",
                "source": "Eurostat",
                "code": "nrg_pc_205",
                "purpose": "SC1/SC2 power cost dimension",
                "status": "to_download",
                "local_path": "data/raw/eurostat/nrg_pc_205.csv",
            },
            {
                "id": "D02",
                "label": "Renewable electricity share by country",
                "source": "Eurostat",
                "code": "nrg_ind_ren",
                "purpose": "SC4 energy transition profile",
                "status": "to_download",
                "local_path": "data/raw/eurostat/nrg_ind_ren.csv",
            },
            {
                "id": "D03",
                "label": "Grid cross-border NTC capacity",
                "source": "ENTSO-E Transparency Platform",
                "code": "Cross-Border Physical Flows > NTC",
                "purpose": "SC1/SC4 grid interconnect capacity",
                "status": "to_download",
                "local_path": "data/raw/entso_e/ntc_capacity.csv",
            },
            {
                "id": "D04",
                "label": "Water Exploitation Index (WEI2)",
                "source": "EEA",
                "code": "WEI2 country/basin level",
                "purpose": "SC1 cooling water availability",
                "status": "to_download",
                "local_path": "data/raw/manual/eea_wei2.csv",
            },
            {
                "id": "D05",
                "label": "Rail freight connectivity",
                "source": "Eurostat",
                "code": "tran_r_rago",
                "purpose": "SC2 logistics / inland freight access",
                "status": "to_download",
                "local_path": "data/raw/eurostat/tran_r_rago.csv",
            },
            {
                "id": "D06",
                "label": "Port throughput (nearest-port proxy)",
                "source": "Eurostat",
                "code": "mar_go_aa",
                "purpose": "SC2 maritime logistics",
                "status": "to_download",
                "local_path": "data/raw/eurostat/mar_go_aa.csv",
            },
            {
                "id": "D07",
                "label": "GERD by sector (incl. DEF)",
                "source": "Eurostat",
                "code": "rd_e_gerdsc",
                "purpose": "SC3 academic R&D / SC5 defence R&D",
                "status": "to_download",
                "local_path": "data/raw/eurostat/rd_e_gerdsc.csv",
            },
            {
                "id": "D08",
                "label": "EU Structural Funds absorption rate",
                "source": "EC Open Data / Cohesion Data",
                "code": "esif_2014-2020_categorisation_ERDF-ESF-CF_planned_vs_achieved",
                "purpose": "SC4 transition readiness / EU investment signal",
                "status": "to_download",
                "local_path": "data/raw/manual/esif_absorption.csv",
            },
            {
                "id": "D09",
                "label": "Ease of doing business index",
                "source": "World Bank",
                "code": "Doing Business 2020 (DB2020)",
                "purpose": "All site classes — regulatory environment",
                "status": "to_download",
                "local_path": "data/raw/world_bank/doing_business_2020.csv",
            },
            {
                "id": "D10",
                "label": "Corporate tax rate (headline + OECD effective)",
                "source": "OECD Tax Database",
                "code": "Table II.1",
                "purpose": "All site classes — fiscal environment",
                "status": "to_download",
                "local_path": "data/raw/oecd/corporate_tax_rates.csv",
            },
            {
                "id": "D11",
                "label": "Broadband ultra-fast coverage (>100 Mbps)",
                "source": "Eurostat",
                "code": "isoc_r_broad_h",
                "purpose": "SC1/SC3 digital infrastructure depth",
                "status": "to_download",
                "local_path": "data/raw/eurostat/isoc_r_broad_h.csv",
            },
            {
                "id": "D12",
                "label": "Corine Land Cover 2018 (industrial land proxy)",
                "source": "Copernicus / EEA",
                "code": "CLC2018",
                "purpose": "SC2 industrial land availability",
                "status": "to_download",
                "local_path": "data/raw/manual/clc2018_nuts2_agg.csv",
            },
        ],
        "disruptor_types": [
            {"id": t, "label": DISRUPTOR_TYPE_LABELS[t]} for t in DISRUPTOR_TYPES
        ],
        "data_gap_proxies": {
            "quantum_patents_nuts2": "DATA_GAP — proxy: total EPO patent density + EuroHPC proximity + national quantum programme flag",
            "robotics_patents_nuts2": "DATA_GAP — proxy: machinery/automotive LQ + BERD + CORDIS EIC funding",
            "haleu_patents_nuts2": "DATA_GAP — proxy: nuclear capacity proximity + chemical LQ (C20) + regulatory flag",
            "grid_headroom_nuts2": "DATA_GAP — proxy: ENTSO-E NTC country capacity + electricity price (inverse)",
            "industrial_land_cost_nuts2": "DATA_GAP — proxy: inverse GDP/cap PPS + urban/rural typology",
        },
    }
    out = ROOT / "analysis" / "p0_data_catalog.json"
    out.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> None:
    with pipeline_step("p0_structure_check"):
        failures = check_structure()

    with pipeline_step("p0_upstream_gold"):
        gold_info = check_upstream_gold()

    with pipeline_step("p0_catalog_stub"):
        catalog_path = write_catalog_stub()

    print()
    print("=" * 65)
    print("P0 SCAFFOLD GATE CHECK")
    print("=" * 65)

    dir_ok = sum(1 for f in failures if "DIR" in f)
    file_ok = sum(1 for f in failures if "FILE" in f)
    dirs_pass = dir_ok == 0
    files_pass = file_ok == 0

    print(f"  Directories    : {'PASS' if dirs_pass else f'FAIL ({dir_ok} missing)'}")
    print(f"  Required files : {'PASS' if files_pass else f'FAIL ({file_ok} missing)'}")
    print(f"  Upstream Gold  : {'PASS' if gold_info['available'] else 'WARN (not found)'}")
    if gold_info["available"]:
        print(f"    shape={gold_info['shape']}  index={gold_info['index_name']}")
    else:
        print(f"    expected at: {gold_info['path']}")
    print(f"  Catalog stub   : {catalog_path.relative_to(ROOT)}")
    print(f"  Disruptor types: {', '.join(DISRUPTOR_TYPES)} (8 types; k=4 clustering override)")

    if failures:
        print()
        print("  Failures:")
        for f in failures:
            print(f"    {f}")

    all_pass = dirs_pass and files_pass
    print()
    print(f"GATE_P0={'PASS' if all_pass else 'FAIL'}")
    print("=" * 65)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
