"""
P11b -- Institutional & Governance Extension.

Sources (see docs/superpowers/specs/2026-09-26-megacampus-p11b-governance-design.md):
  smart_spec_strategy_flag       -- JRC S3 Platform strategy registry
  cluster_org_count              -- European Cluster Collaboration Platform registry
  institutional_diversity_score  -- Eurostat educ_uoe_enrt01 (tertiary institutions)
  rda_capacity_index             -- ESIF dashboard bulk CSV (active co-financed programmes)
  regional_fiscal_autonomy_pct   -- OECD Fiscal Decentralisation Database (country-level)

Purely additive to Gold -- never consumed by p4_suitability_scores.py.
Real data only: a fetcher that cannot retrieve its source returns False and
the harmonisation step records a DATA GAP with null + imputed_flag=True,
rather than fabricating a value.
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

np.random.seed(42)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logging_config import pipeline_step  # noqa: E402

RAW = ROOT / "data" / "raw"
GOLD_PATH = ROOT / "data" / "gold" / "megacampus_gold.parquet"
GOVERNANCE_GOLD_PATH = ROOT / "data" / "gold" / "megacampus_governance.parquet"
ANALYSIS = ROOT / "analysis"
TIMEOUT = 60
RETRY_WAIT = 5


def _get(url: str, params: dict | None = None, stream: bool = False) -> requests.Response:
    headers = {"User-Agent": "EU-MegaCampus-Research/1.0 (academic; non-commercial)"}
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=TIMEOUT, stream=stream)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(RETRY_WAIT * (attempt + 1))
    raise RuntimeError("unreachable")


def _skip_or_force(path: Path, force: bool) -> bool:
    if path.exists() and not force:
        print(f"    SKIP (exists): {path.relative_to(ROOT)}")
        return True
    return False


def fetch_esif_programmes(force: bool) -> bool:
    """ESIF dashboard public bulk CSV -- active co-financed programmes by NUTS2."""
    out_path = RAW / "esif" / "esif_programmes.csv"
    if _skip_or_force(out_path, force):
        return True
    try:
        url = "https://cohesiondata.ec.europa.eu/api/views/e4v6-qrrs/rows.csv"
        r = _get(url, params={"accessType": "DOWNLOAD"})
        df = pd.read_csv(io.StringIO(r.text))
        if df.shape[0] < 100:
            raise ValueError(f"suspiciously small response ({df.shape[0]} rows) -- likely an error page, not the dataset")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(r.content)
        print(f"    OK  esif_programmes: {df.shape[0]:,} rows -> {out_path.relative_to(ROOT)}")
        return True
    except Exception as e:
        print(f"    FAIL esif_programmes: {e}")
        return False


def fetch_educ_institutions(force: bool) -> bool:
    """Eurostat educ_uoe_enrt01 -- used as a proxy for institutional diversity."""
    out_path = RAW / "eurostat" / "educ_uoe_enrt01.csv"
    if _skip_or_force(out_path, force):
        return True
    try:
        base = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data"
        r = _get(f"{base}/educ_uoe_enrt01", params={"format": "SDMX-CSV", "lang": "EN", "sinceTimePeriod": "2018"})
        df = pd.read_csv(io.StringIO(r.text))
        if df.shape[0] < 50:
            raise ValueError(f"suspiciously small response ({df.shape[0]} rows)")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(r.content)
        print(f"    OK  educ_uoe_enrt01: {df.shape[0]:,} rows -> {out_path.relative_to(ROOT)}")
        return True
    except Exception as e:
        print(f"    FAIL educ_uoe_enrt01: {e}")
        return False


def fetch_oecd_fiscal_autonomy(force: bool) -> bool:
    """OECD Fiscal Decentralisation Database -- country-level own-revenue share.

    No confirmed bulk-CSV endpoint at plan-writing time; this fetcher tries
    the OECD.Stat SDMX-JSON API and falls back to reporting the gap honestly
    rather than guessing an endpoint shape that may not exist.
    """
    out_path = RAW / "oecd" / "fiscal_decentralisation.csv"
    if _skip_or_force(out_path, force):
        return True
    try:
        url = "https://sdmx.oecd.org/public/rest/data/OECD.CTP.FBT,DSD_FISCAL_DECENTRALISATION@DF_FD,1.0/all"
        r = _get(url, params={"format": "csvfilewithlabels"})
        df = pd.read_csv(io.StringIO(r.text))
        if df.shape[0] < 10:
            raise ValueError(f"suspiciously small response ({df.shape[0]} rows)")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(r.content)
        print(f"    OK  fiscal_decentralisation: {df.shape[0]:,} rows -> {out_path.relative_to(ROOT)}")
        return True
    except Exception as e:
        print(f"    FAIL fiscal_decentralisation (DATA GAP, will be recorded): {e}")
        return False


def fetch_s3_strategies(force: bool) -> bool:
    """JRC S3 Platform -- registered smart specialisation strategies."""
    out_path = RAW / "jrc" / "s3_strategies.csv"
    if _skip_or_force(out_path, force):
        return True
    try:
        url = "https://s3platform.jrc.ec.europa.eu/s3-community-of-practice/-/asset_publisher/rTB2sZTVOOlB/document/export"
        r = _get(url)
        df = pd.read_csv(io.StringIO(r.text))
        if df.shape[0] < 10:
            raise ValueError(f"suspiciously small response ({df.shape[0]} rows)")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(r.content)
        print(f"    OK  s3_strategies: {df.shape[0]:,} rows -> {out_path.relative_to(ROOT)}")
        return True
    except Exception as e:
        print(f"    FAIL s3_strategies (DATA GAP, will be recorded): {e}")
        return False


def fetch_cluster_registry(force: bool) -> bool:
    """European Cluster Collaboration Platform -- registered cluster organisations."""
    out_path = RAW / "ecc" / "cluster_registry.csv"
    if _skip_or_force(out_path, force):
        return True
    try:
        url = "https://clustercollaboration.eu/system/files/documents/cluster_organisations_export.csv"
        r = _get(url)
        df = pd.read_csv(io.StringIO(r.text))
        if df.shape[0] < 10:
            raise ValueError(f"suspiciously small response ({df.shape[0]} rows)")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(r.content)
        print(f"    OK  cluster_registry: {df.shape[0]:,} rows -> {out_path.relative_to(ROOT)}")
        return True
    except Exception as e:
        print(f"    FAIL cluster_registry (DATA GAP, will be recorded): {e}")
        return False
