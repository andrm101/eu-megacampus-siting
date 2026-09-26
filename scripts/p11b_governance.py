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


def load_gold_keys() -> pd.DataFrame:
    gold = pd.read_parquet(GOLD_PATH, columns=["country_code"])
    return gold


def broadcast_country_to_nuts2(country_values: pd.Series, gold_keys: pd.DataFrame) -> pd.Series:
    mapped = gold_keys["country_code"].map(country_values)
    mapped.index = gold_keys.index
    mapped.name = None
    return mapped


def _parse_esif_programmes(gold_keys: pd.DataFrame) -> pd.Series | None:
    path = RAW / "esif" / "esif_programmes.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    nuts_col = next((c for c in df.columns if "nuts" in c), None)
    if nuts_col is None:
        return None
    df["nuts2"] = df[nuts_col].astype(str).str.upper().str[:4]
    counts = df.groupby("nuts2").size()
    counts = counts[counts.index.isin(gold_keys.index)]
    return counts.reindex(gold_keys.index)


def _parse_s3_strategies(gold_keys: pd.DataFrame) -> pd.Series | None:
    path = RAW / "jrc" / "s3_strategies.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    country_col = next((c for c in df.columns if "countr" in c), None)
    if country_col is None:
        return None
    countries_with_strategy = set(df[country_col].astype(str).str.upper().str[:2])
    country_flag = pd.Series(
        {c: (c in countries_with_strategy) for c in gold_keys["country_code"].unique()}
    )
    return broadcast_country_to_nuts2(country_flag, gold_keys)


def _parse_cluster_registry(gold_keys: pd.DataFrame) -> pd.Series | None:
    path = RAW / "ecc" / "cluster_registry.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    nuts_col = next((c for c in df.columns if "nuts" in c), None)
    if nuts_col is None:
        return None
    df["nuts2"] = df[nuts_col].astype(str).str.upper().str[:4]
    counts = df.groupby("nuts2").size()
    counts = counts[counts.index.isin(gold_keys.index)]
    return counts.reindex(gold_keys.index)


def _parse_educ_institutions(gold_keys: pd.DataFrame) -> pd.Series | None:
    path = RAW / "eurostat" / "educ_uoe_enrt01.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    geo_col = next((c for c in df.columns if c == "geo"), None)
    val_col = "obs_value" if "obs_value" in df.columns else None
    if geo_col is None or val_col is None:
        return None
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df["geo4"] = df[geo_col].astype(str).str.upper().str[:4]
    nuts2_rows = df[df["geo4"].isin(gold_keys.index)]
    if len(nuts2_rows) < 20:
        # too sparse at NUTS2 to trust -- degrade to country-level broadcast
        df["geo2"] = df[geo_col].astype(str).str.upper().str[:2]
        country_level = df[df["geo2"].str.len() == 2].groupby("geo2")[val_col].mean()
        return broadcast_country_to_nuts2(country_level, gold_keys)
    return nuts2_rows.groupby("geo4")[val_col].mean().reindex(gold_keys.index)


def _parse_oecd_fiscal_autonomy(gold_keys: pd.DataFrame) -> pd.Series | None:
    path = RAW / "oecd" / "fiscal_decentralisation.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    country_col = next((c for c in df.columns if "country" in c or c == "location"), None)
    val_col = next((c for c in df.columns if "value" in c or "obs" in c), None)
    if country_col is None or val_col is None:
        return None
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    country_level = df.groupby(df[country_col].astype(str).str.upper().str[:2])[val_col].mean()
    return broadcast_country_to_nuts2(country_level, gold_keys)


def harmonise_governance(gold_keys: pd.DataFrame, fetch_results: dict[str, bool]) -> tuple[pd.DataFrame, dict]:
    out = pd.DataFrame(index=gold_keys.index)
    gap_report = {}

    def _add(col: str, series: pd.Series | None, fetch_key: str, broadcast_flag_name: str | None = None):
        if series is None:
            out[col] = np.nan
            flag_col = broadcast_flag_name or f"{col}_imputed_flag"
            out[flag_col] = True
            gap_report[col] = {
                "status": "DATA_GAP",
                "reason": f"source '{fetch_key}' unavailable this session (fetch_results={fetch_results.get(fetch_key)})",
                "n_null": len(out),
            }
            return
        out[col] = series.values
        flag_col = broadcast_flag_name or f"{col}_imputed_flag"
        out[flag_col] = series.isna().values
        n_null = int(series.isna().sum())
        gap_report[col] = {
            "status": "OK" if n_null == 0 else "PARTIAL_DATA_GAP",
            "reason": f"{n_null}/{len(series)} regions unresolved" if n_null else "fully resolved",
            "n_null": n_null,
        }

    _add("smart_spec_strategy_flag", _parse_s3_strategies(gold_keys), "s3_strategies")
    _add("cluster_org_count", _parse_cluster_registry(gold_keys), "cluster_registry")
    _add("institutional_diversity_score", _parse_educ_institutions(gold_keys), "educ_institutions")
    _add("rda_capacity_index", _parse_esif_programmes(gold_keys), "esif_programmes")
    _add(
        "regional_fiscal_autonomy_pct",
        _parse_oecd_fiscal_autonomy(gold_keys),
        "oecd_fiscal_autonomy",
        broadcast_flag_name="regional_fiscal_autonomy_is_country_broadcast",
    )
    # fiscal autonomy is *always* a country broadcast by design when present,
    # not only when null -- overwrite the flag to reflect that regardless of
    # the generic isna()-based default _add() just set.
    if "regional_fiscal_autonomy_pct" in out.columns:
        out["regional_fiscal_autonomy_is_country_broadcast"] = out["regional_fiscal_autonomy_pct"].notna()

    return out, gap_report


def append_to_gold(governance_df: pd.DataFrame) -> dict:
    gold = pd.read_parquet(GOLD_PATH)
    pre_existing_cols = [c for c in gold.columns if c not in governance_df.columns]
    hash_before = _hash_columns(gold, pre_existing_cols)

    merged = gold.copy()
    for col in governance_df.columns:
        merged[col] = governance_df[col]

    hash_after = _hash_columns(merged[pre_existing_cols], pre_existing_cols)

    merged.to_parquet(GOLD_PATH)
    governance_df.to_parquet(GOVERNANCE_GOLD_PATH)

    return {
        "pre_existing_columns_hash_before": hash_before,
        "pre_existing_columns_hash_after": hash_after,
        "unchanged": hash_before == hash_after,
    }


def _hash_columns(df: pd.DataFrame, cols: list[str]) -> str:
    import hashlib
    content = df[cols].to_csv(index=True).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def run_p4_rerun_check() -> dict:
    """Re-run p4_suitability_scores.py and confirm T1-T8 counts match the
    pre-P11b baseline recorded in CLAUDE.md -- the concrete proof this
    extension changed nothing about scoring, not just an inspection claim."""
    import re
    import subprocess

    baseline = {"T1": 49, "T2": 46, "T3": 75, "T4": 77, "T5": 21, "T6": 38, "T7": 31, "T8": 106}
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "p4_suitability_scores.py")],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    counts = {}
    for line in result.stdout.splitlines():
        m = re.search(r"\b(T[1-8])\D+(\d+)\s*$", line)
        if m:
            counts[m.group(1)] = int(m.group(2))
    matches = {t: counts.get(t) == baseline[t] for t in baseline}
    return {
        "status": "PASS" if all(matches.values()) else "FAIL",
        "baseline": baseline,
        "observed": counts,
        "matches": matches,
    }


def run_gate_p11b(governance_df: pd.DataFrame, gap_report: dict, hash_check: dict,
                   expected_n_regions: int = 242, p4_rerun: dict | None = None) -> dict:
    report = {}

    n_rows = len(governance_df)
    n_unique = governance_df.index.nunique()
    report["row_count"] = {
        "status": "PASS" if (n_rows == expected_n_regions and n_unique == expected_n_regions) else "FAIL",
        "n_rows": n_rows, "n_unique": n_unique, "expected": expected_n_regions,
    }

    required_cols = {
        "smart_spec_strategy_flag", "smart_spec_strategy_flag_imputed_flag",
        "cluster_org_count", "cluster_org_count_imputed_flag",
        "institutional_diversity_score", "institutional_diversity_score_imputed_flag",
        "rda_capacity_index", "rda_capacity_index_imputed_flag",
        "regional_fiscal_autonomy_pct", "regional_fiscal_autonomy_is_country_broadcast",
    }
    missing = required_cols - set(governance_df.columns)
    report["columns_present"] = {"status": "PASS" if not missing else "FAIL", "missing": sorted(missing)}

    report["gold_unchanged"] = {"status": "PASS" if hash_check.get("unchanged") else "FAIL", **hash_check}

    report["data_gap_register"] = {"status": "PASS", "gaps": gap_report}

    if p4_rerun is not None:
        report["p4_rerun_unchanged"] = p4_rerun

    overall = all(v.get("status") == "PASS" for v in report.values())
    report["overall_status"] = "PASS" if overall else "FAIL"
    return report


def main(force: bool = False, skip_p4_rerun: bool = False) -> None:
    with pipeline_step("p11b_governance", input_artifact=GOLD_PATH) as log:
        print("Fetching governance sources...")
        fetch_results = {
            "s3_strategies": fetch_s3_strategies(force),
            "cluster_registry": fetch_cluster_registry(force),
            "educ_institutions": fetch_educ_institutions(force),
            "esif_programmes": fetch_esif_programmes(force),
            "oecd_fiscal_autonomy": fetch_oecd_fiscal_autonomy(force),
        }
        log.info("fetch_complete", results=fetch_results)

        gold_keys = load_gold_keys()
        governance_df, gap_report = harmonise_governance(gold_keys, fetch_results)

        hash_check = append_to_gold(governance_df)

        p4_rerun = None if skip_p4_rerun else run_p4_rerun_check()

        gate = run_gate_p11b(governance_df, gap_report, hash_check, p4_rerun=p4_rerun)

        ANALYSIS.mkdir(exist_ok=True)
        (ANALYSIS / "p11b_governance_report.json").write_text(json.dumps(gate, indent=2, default=str))

        print(json.dumps(gate, indent=2, default=str))
        print(f"\nGATE_P11b={gate['overall_status']}")
        log.info("step_complete", gate=gate["overall_status"])
        sys.exit(0 if gate["overall_status"] == "PASS" else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-p4-rerun", action="store_true",
                         help="Skip the P4 subprocess re-run (useful for fast local iteration)")
    args = parser.parse_args()
    main(force=args.force, skip_p4_rerun=args.skip_p4_rerun)
