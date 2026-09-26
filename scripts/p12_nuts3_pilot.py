"""
P12 -- NUTS3 Down-Scaling Pilot (Optional/future phase).

Down-scales 3 Tier-1 NUTS2 regions (SE11 Stockholm, FI1B Helsinki-Uusimaa,
DK01 Hovedstaden) to their NUTS3 children for T5/T7 suitability, per
docs/superpowers/specs/2026-09-26-p12-nuts3-downscaling-design.md.

Standalone output only -- never modifies megacampus_gold.parquet or any
existing gate artifact. Of T5/T7's 15 unique weighted features, 3
attempt genuine NUTS3 resolution (artificial_land_pct, epo_patents_per_mio_pop,
lq_nace_c26 via a NACE-section proxy); the rest broadcast the parent
NUTS2's existing Gold value with an honest *_is_nuts2_broadcast flag.
"""

import argparse
import hashlib
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
OUT_PATH = ROOT / "data" / "gold" / "p12_nuts3_pilot.parquet"
ANALYSIS = ROOT / "analysis"
TIMEOUT = 60
RETRY_WAIT = 5

PILOT_NUTS2 = ["SE11", "FI1B", "DK01"]


def _get(url: str, params: dict | None = None) -> requests.Response:
    headers = {"User-Agent": "EU-MegaCampus-Research/1.0 (academic; non-commercial)"}
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(RETRY_WAIT * (attempt + 1))
    raise RuntimeError("unreachable")


def _skip_or_force(path: Path, force: bool) -> bool:
    if path.exists() and not force:
        try:
            shown = path.relative_to(ROOT)
        except ValueError:
            shown = path
        print(f"    SKIP (exists): {shown}")
        return True
    return False


def _fetch_eurostat_dataset(code: str, out_path: Path, force: bool) -> bool:
    if _skip_or_force(out_path, force):
        return True
    try:
        base = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data"
        r = _get(f"{base}/{code}", params={"format": "SDMX-CSV", "lang": "EN"})
        df = pd.read_csv(io.StringIO(r.text))
        if df.shape[0] < 20:
            raise ValueError(f"suspiciously small response ({df.shape[0]} rows)")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(r.content)
        print(f"    OK  {code}: {df.shape[0]:,} rows -> {out_path.relative_to(ROOT)}")
        return True
    except Exception as e:
        print(f"    FAIL {code}: {e}")
        return False


def fetch_reg_area3(force: bool) -> bool:
    return _fetch_eurostat_dataset("reg_area3", RAW / "eurostat" / "reg_area3.csv", force)


def fetch_pat_ep_rtot(force: bool) -> bool:
    return _fetch_eurostat_dataset("pat_ep_rtot", RAW / "eurostat" / "pat_ep_rtot.csv", force)


def fetch_nama_10r_3empers(force: bool) -> bool:
    """nama_10r_3empers's full EU-wide bulk file is ~111MB (over GitHub's
    100MB limit) but this pilot only ever needs 9 geo codes (3 pilot
    parents + their country totals) x wstatus=='EMP' x nace_r2 in
    {'C','TOTAL'} -- filtered down to ~450 rows immediately after fetching,
    before writing to disk, so a --force re-run never reintroduces the
    oversized file (caught in final review: an earlier version committed
    the full 111MB dump)."""
    out_path = RAW / "eurostat" / "nama_10r_3empers.csv"
    if _skip_or_force(out_path, force):
        return True
    try:
        base = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data"
        r = _get(f"{base}/nama_10r_3empers", params={"format": "SDMX-CSV", "lang": "EN"})
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.lower().strip() for c in df.columns]
        if df.shape[0] < 20:
            raise ValueError(f"suspiciously small response ({df.shape[0]} rows)")

        country_codes = {p[:2] for p in PILOT_NUTS2}
        geo = df["geo"].astype(str).str.upper()
        is_pilot_nuts3 = (geo.str.len() == 5) & (geo.str[:4].isin(PILOT_NUTS2))
        keep_geo = is_pilot_nuts3 | geo.isin(country_codes)
        filtered = df[keep_geo & (df["wstatus"] == "EMP") & (df["nace_r2"].isin(["C", "TOTAL"]))]
        if filtered.shape[0] < 10:
            raise ValueError(f"filtered result suspiciously small ({filtered.shape[0]} rows) -- check geo/wstatus/nace filters")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        filtered.to_csv(out_path, index=False)
        print(f"    OK  nama_10r_3empers: {df.shape[0]:,} rows fetched, {filtered.shape[0]:,} kept -> {out_path.relative_to(ROOT)}")
        return True
    except Exception as e:
        print(f"    FAIL nama_10r_3empers: {e}")
        return False


# ─── T5/T7 formula specs, copied verbatim from p4_suitability_scores.py ──────

T5_SPECS: list[tuple[str, int, str]] = [
    ("electricity_price_eur_kwh", 3, "-"),
    ("ntc_import_mw", 3, "+"),
    ("renewable_energy_share_pct", 3, "+"),
    ("ultrafast_broadband_pct", 3, "+"),
    ("water_exploitation_index", 2, "-"),
    ("artificial_land_pct", 2, "+"),
    ("doing_business_score", 2, "+"),
    ("score_infra", 1, "+"),
]

T7_SPECS: list[tuple[str, int, str]] = [
    ("lq_nace_c26", 3, "+"),
    ("score_talent", 3, "+"),
    ("hrst_per_1000", 2, "+"),
    ("gerd_hes_pct_gdp", 2, "+"),
    ("epo_patents_per_mio_pop", 2, "+"),
    ("score_cluster", 2, "+"),
    ("doing_business_score", 2, "+"),
    ("rail_freight_ktonnes", 1, "+"),
]

# Features attempting genuine NUTS3 resolution, per the spec's resolvability table.
GENUINE_RESOLUTION_FEATURES = {"artificial_land_pct", "epo_patents_per_mio_pop", "lq_nace_c26"}


def _validate_child_counts(nuts3_df: pd.DataFrame, pilot_parents: list[str], min_children: int = 2) -> None:
    counts = nuts3_df["parent_nuts2"].value_counts()
    for parent in pilot_parents:
        n = int(counts.get(parent, 0))
        if n < min_children:
            raise ValueError(
                f"pilot parent '{parent}' has only {n} NUTS3 children in the derived "
                f"geo-code set -- expected at least {min_children}; check NUTS vintage/parsing"
            )


def resolve_nuts3_children(min_children: int = 1) -> pd.DataFrame:
    """Derive each pilot NUTS2's NUTS3 children from the geo codes actually
    present in the fetched Eurostat datasets (reg_area3 + pat_ep_rtot +
    nama_10r_3empers), rather than a separate correspondence-table fetch -- Eurostat's
    geodata/nuts bulk-CSV endpoint guessed for that table 404s, and the
    datasets we already have contain the real NUTS3 codes directly.

    min_children defaults to 1, not 2: confirmed independently against two
    Eurostat datasets that SE11 (Stockholm) and FI1B (Helsinki-Uusimaa) each
    have exactly ONE NUTS3 child in the current nomenclature -- a genuine
    geographic fact (small/single-county NUTS2 regions collapsing to one
    NUTS3 child is common in this classification), not a data gap. Only
    DK01 (Hovedstaden) has multiple children (4)."""
    codes = set()
    for fname in ("reg_area3.csv", "pat_ep_rtot.csv", "nama_10r_3empers.csv"):
        path = RAW / "eurostat" / fname
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.lower().strip() for c in df.columns]
        if "geo" not in df.columns:
            continue
        geo = df["geo"].astype(str).str.upper()
        codes |= set(geo[geo.str.len() == 5])

    rows = [(c, c[:4]) for c in sorted(codes) if c[:4] in PILOT_NUTS2]
    df = pd.DataFrame(rows, columns=["nuts3_code", "parent_nuts2"]).set_index("nuts3_code")
    _validate_child_counts(df.reset_index(), PILOT_NUTS2, min_children=min_children)
    return df


def _parse_reg_area3(nuts3_children: pd.DataFrame) -> pd.Series | None:
    """artificial_land_pct: reg_area3's only two landuse codes are L0008
    (land area, i.e. total area minus inland water) and TOTAL (including
    water). L0008/TOTAL is consistently ~93-98% for every region checked --
    that is a land-vs-water ratio, not an artificial/built-up land share.
    There is no genuine artificial-land-percentage breakdown in this
    dataset, so this always degrades to BROADCAST (verified against real
    Eurostat data during Task 3's live run and the final whole-branch
    review, which caught an earlier version of this function silently
    reporting land-vs-water ratios as if they were artificial-land %)."""
    return None


def _parse_pat_ep_rtot(nuts3_children: pd.DataFrame) -> pd.Series | None:
    """epo_patents_per_mio_pop: pat_ep_rtot mixes 4 incompatible unit
    dimensions (NR counts, P_MHAB per-million-inhabitants, GDP_BEUR,
    P_MACT) across years 1977-2012. Must filter to unit=='P_MHAB' (the
    per-capita measure this column name promises) and use the latest
    available year per region -- an earlier version of this function
    averaged obs_value across all four units and all years, producing a
    meaningless blended magnitude (caught in final review)."""
    path = RAW / "eurostat" / "pat_ep_rtot.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.lower().strip() for c in df.columns]
    if not {"geo", "obs_value", "unit", "time_period"}.issubset(df.columns):
        return None
    df = df[df["unit"] == "P_MHAB"]
    df["obs_value"] = pd.to_numeric(df["obs_value"], errors="coerce")
    df = df[df["geo"].isin(nuts3_children.index)]
    if len(df) < len(nuts3_children) // 2:
        return None  # coverage too sparse to trust as genuine
    latest = df.sort_values("time_period").groupby("geo").tail(1)
    return latest.set_index("geo")["obs_value"].reindex(nuts3_children.index)


def _parse_nama_10r_3empers_as_lq_proxy(nuts3_children: pd.DataFrame) -> pd.Series | None:
    """lq_nace_c26 proxy: NUTS3 doesn't publish 2-digit NACE (C26) detail, so
    this computes a genuine location quotient using NACE section C
    (manufacturing) as the closest available substitute:

        LQ = (region_C_employment / region_TOTAL_employment)
           / (country_C_employment / country_TOTAL_employment)

    This is a real ratio, coarser than NUTS2's 2-digit measurement but
    still a location quotient -- an earlier version of this function
    returned raw manufacturing headcount in thousands (an absolute
    magnitude, not a quotient at all; caught in final review). Uses
    wstatus=='EMP' (employed persons) and the latest common year."""
    path = RAW / "eurostat" / "nama_10r_3empers.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.lower().strip() for c in df.columns]
    required = {"geo", "obs_value", "wstatus", "nace_r2", "time_period"}
    if not required.issubset(df.columns):
        return None
    df = df[df["wstatus"] == "EMP"]
    df["obs_value"] = pd.to_numeric(df["obs_value"], errors="coerce")
    df["geo"] = df["geo"].astype(str).str.upper()

    latest_year = df[df["geo"].isin(nuts3_children.index)]["time_period"].max()
    if pd.isna(latest_year):
        return None
    df = df[df["time_period"] == latest_year]

    def _share(geo_codes: set, nace: str) -> pd.Series:
        sub = df[df["geo"].isin(geo_codes) & (df["nace_r2"] == nace)]
        return sub.set_index("geo")["obs_value"]

    region_codes = set(nuts3_children.index)
    region_c = _share(region_codes, "C")
    region_total = _share(region_codes, "TOTAL")
    if len(region_c) < len(nuts3_children) // 2 or len(region_total) < len(nuts3_children) // 2:
        return None
    region_share = (region_c / region_total).reindex(nuts3_children.index)

    country_codes = set(nuts3_children["parent_nuts2"].str[:2].unique())
    country_c = _share(country_codes, "C")
    country_total = _share(country_codes, "TOTAL")
    country_share = (country_c / country_total)
    if country_share.empty:
        return None
    country_share_by_parent = nuts3_children["parent_nuts2"].str[:2].map(country_share)

    return region_share / country_share_by_parent


def _broadcast_from_gold(feature: str, nuts3_children: pd.DataFrame) -> pd.Series:
    gold_full = pd.read_parquet(GOLD_PATH)
    if feature not in gold_full.columns:
        return pd.Series(float("nan"), index=nuts3_children.index)
    parent_values = gold_full.loc[gold_full.index.isin(PILOT_NUTS2), feature]
    return nuts3_children["parent_nuts2"].map(parent_values)


def build_nuts3_panel(nuts3_children: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = pd.DataFrame(index=nuts3_children.index)
    out["parent_nuts2"] = nuts3_children["parent_nuts2"]
    report = {}

    genuine_parsers = {
        "artificial_land_pct": _parse_reg_area3,
        "epo_patents_per_mio_pop": _parse_pat_ep_rtot,
        "lq_nace_c26": _parse_nama_10r_3empers_as_lq_proxy,
    }

    all_features = {f for f, _, _ in T5_SPECS} | {f for f, _, _ in T7_SPECS}

    # lq_nace_c26 is a genuine ratio but a coarser NACE-section proxy, not
    # NUTS2's 2-digit measurement -- flagged distinctly (status "PROXY", not
    # "GENUINE") so a reader of the parquet/gate report alone can tell the
    # two are not computed the same way, per the final review's finding.
    proxy_features = {"lq_nace_c26"}

    for feat in all_features:
        if feat in GENUINE_RESOLUTION_FEATURES:
            series = genuine_parsers[feat](nuts3_children)
            if series is not None and series.notna().sum() >= len(nuts3_children) // 2:
                out[feat] = series
                out[f"{feat}_is_nuts2_broadcast"] = False
                if feat in proxy_features:
                    out[f"{feat}_is_section_proxy"] = True
                    report[feat] = {"status": "PROXY", "reason": "NACE-section-C location quotient, not the exact 2-digit NUTS2 measurement"}
                else:
                    report[feat] = {"status": "GENUINE", "reason": "resolved at NUTS3"}
                continue
            report[feat] = {
                "status": "GENUINE_FAILED_SANITY_CHECK" if series is not None else "BROADCAST",
                "reason": "genuine source unavailable or too sparse -- degraded to broadcast",
            }

        out[feat] = _broadcast_from_gold(feat, nuts3_children)
        out[f"{feat}_is_nuts2_broadcast"] = True
        if feat in proxy_features:
            out[f"{feat}_is_section_proxy"] = False
        if feat not in report:
            report[feat] = {"status": "BROADCAST", "reason": "no NUTS3 source per spec's resolvability table"}

    return out, report


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def compute_composite(panel: pd.DataFrame, specs: list[tuple[str, int, str]]) -> pd.Series:
    total_w = sum(w for _, w, _ in specs)
    weighted = pd.Series(0.0, index=panel.index)
    for feat, w, direction in specs:
        vals = panel[feat].fillna(panel[feat].median())
        norm = _minmax(vals)
        if direction == "-":
            norm = 1.0 - norm
        weighted += norm * w
    return weighted / total_w


def _get_parent_gold_value(feature: str, parent: str) -> float | None:
    gold_full = pd.read_parquet(GOLD_PATH)
    if feature not in gold_full.columns or parent not in gold_full.index:
        return None
    return float(gold_full.loc[parent, feature])


def compute_sanity_check(panel: pd.DataFrame, report: dict) -> dict:
    """Spec's Gate criterion 3: a documented consistency figure (not a hard
    pass/fail gate) comparing each genuinely-resolved (GENUINE or PROXY)
    feature's NUTS3 mean against its parent NUTS2's existing Gold value.
    For SE11/FI1B (single NUTS3 child each), this is close to an identity
    check; for DK01 (4 children), it's a real aggregation consistency
    figure. Silently dropped from an earlier version of this script --
    caught in final review, since it would have surfaced the unit-mismatch
    bugs in the three parsers immediately."""
    sanity = {}
    genuine_features = [f for f, v in report.items() if v["status"] in ("GENUINE", "PROXY")]
    for feat in genuine_features:
        sanity[feat] = {}
        for parent, grp in panel.groupby("parent_nuts2"):
            nuts3_mean = float(grp[feat].mean())
            parent_gold = _get_parent_gold_value(feat, parent)
            entry = {"nuts3_mean": nuts3_mean, "parent_gold_value": parent_gold}
            if parent_gold not in (None, 0):
                entry["relative_diff_pct"] = round(100 * abs(nuts3_mean - parent_gold) / abs(parent_gold), 1)
            sanity[feat][parent] = entry
    return sanity


def write_ranking_csv(panel: pd.DataFrame) -> Path:
    rows = []
    for parent, grp in panel.groupby("parent_nuts2"):
        grp = grp.copy()
        grp["T5_rank_within_parent"] = grp["T5_composite"].rank(ascending=False, method="min").astype(int)
        grp["T7_rank_within_parent"] = grp["T7_composite"].rank(ascending=False, method="min").astype(int)
        for nuts3_code, row in grp.iterrows():
            rows.append({
                "parent_nuts2": parent,
                "nuts3_code": nuts3_code,
                "T5_composite": row["T5_composite"],
                "T7_composite": row["T7_composite"],
                "T5_rank_within_parent": row["T5_rank_within_parent"],
                "T7_rank_within_parent": row["T7_rank_within_parent"],
            })
    out = ANALYSIS / "p12_nuts3_ranking.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def _compute_gold_hash() -> str:
    df = pd.read_parquet(GOLD_PATH)
    content = df.to_csv(index=True).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _hash_gold_unchanged(expected_hash: str | None = None) -> dict:
    """Confirm this script never modified megacampus_gold.parquet, by
    comparing its current content hash against a supplied baseline (e.g.
    p10_audit.py's expected_hashes.json entry) -- an earlier version of
    this function always returned unchanged=True unconditionally, without
    comparing against anything, making the gate criterion vacuous by
    construction (caught in final review)."""
    if not GOLD_PATH.exists():
        return {"unchanged": False, "reason": "megacampus_gold.parquet missing"}
    current_hash = _compute_gold_hash()
    if expected_hash is None:
        expected_path = ROOT / "expected_hashes.json"
        if not expected_path.exists():
            return {"unchanged": False, "reason": "no expected_hashes.json baseline to compare against", "hash": current_hash}
        baseline = json.loads(expected_path.read_text())
        expected_hash = baseline.get("data/gold/megacampus_gold.parquet")
        if expected_hash is None:
            return {"unchanged": False, "reason": "megacampus_gold.parquet not in expected_hashes.json", "hash": current_hash}
    return {"unchanged": current_hash == expected_hash, "hash": current_hash, "expected_hash": expected_hash}


def _compute_weight_coverage(report: dict) -> dict:
    """Coverage by weight-points, not feature count -- the spec's Gate
    criterion 2 asks for this, and T5's weights sum to 19 (not 20) and T7's
    to 17 (not 20), both corrected from an earlier draft's wrong assumption."""
    coverage = {}
    for type_id, specs in (("T5", T5_SPECS), ("T7", T7_SPECS)):
        total_weight = sum(w for _, w, _ in specs)
        genuine_or_proxy_weight = sum(
            w for feat, w, _ in specs if report.get(feat, {}).get("status") in ("GENUINE", "PROXY")
        )
        coverage[type_id] = {
            "total_weight": total_weight,
            "genuine_or_proxy_weight": genuine_or_proxy_weight,
            "coverage_pct": round(100 * genuine_or_proxy_weight / total_weight, 1),
        }
    return coverage


def run_gate_p12(panel: pd.DataFrame, report: dict, gold_hash_check: dict) -> dict:
    result = {}

    n_children_per_parent = panel.groupby("parent_nuts2").size()
    # min 1, not 2: SE11 and FI1B genuinely have exactly one NUTS3 child each
    # in the current nomenclature (confirmed against 2 independent Eurostat
    # datasets during Task 3's live run) -- only DK01 has multiple children.
    result["children_present"] = {
        "status": "PASS" if (n_children_per_parent >= 1).all() and set(n_children_per_parent.index) == set(PILOT_NUTS2) else "FAIL",
        "counts": n_children_per_parent.to_dict(),
    }

    n_genuine_or_proxy = sum(1 for v in report.values() if v["status"] in ("GENUINE", "PROXY"))
    result["resolution_report"] = {
        "status": "PASS",
        "n_genuine_or_proxy": n_genuine_or_proxy,
        "n_total": len(report),
        "weight_coverage": _compute_weight_coverage(report),
        "detail": report,
    }

    result["gold_unchanged"] = {
        "status": "PASS" if gold_hash_check.get("unchanged") else "FAIL",
        **gold_hash_check,
    }

    overall = all(v.get("status") == "PASS" for v in result.values())
    result["overall_status"] = "PASS" if overall else "FAIL"
    return result


def main(force: bool = False) -> None:
    with pipeline_step("p12_nuts3_pilot", input_artifact=GOLD_PATH) as log:
        print("Fetching P12 NUTS3 pilot sources...")
        fetch_reg_area3(force)
        fetch_pat_ep_rtot(force)
        fetch_nama_10r_3empers(force)

        nuts3_children = resolve_nuts3_children()
        panel, report = build_nuts3_panel(nuts3_children)
        panel["T5_composite"] = compute_composite(panel, T5_SPECS)
        panel["T7_composite"] = compute_composite(panel, T7_SPECS)

        panel.to_parquet(OUT_PATH)
        write_ranking_csv(panel)

        sanity = compute_sanity_check(panel, report)
        gold_hash_check = _hash_gold_unchanged()
        gate = run_gate_p12(panel, report, gold_hash_check)
        gate["sanity_check"] = sanity

        (ANALYSIS / "p12_gate_report.json").write_text(json.dumps(gate, indent=2, default=str))
        print(json.dumps(gate, indent=2, default=str))
        print(f"\nGATE_P12={gate['overall_status']}")
        log.info("step_complete", gate=gate["overall_status"])
        sys.exit(0 if gate["overall_status"] == "PASS" else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
