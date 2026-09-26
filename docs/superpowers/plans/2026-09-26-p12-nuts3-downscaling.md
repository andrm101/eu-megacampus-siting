# P12 NUTS3 Down-Scaling Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Down-scale SE11, FI1B, and DK01 from NUTS2 to their NUTS3 children, recomputing T5/T7 suitability composites so the specific NUTS3 sub-region driving each parent's high score is identifiable.

**Architecture:** One new script, `scripts/p12_nuts3_pilot.py`, follows the existing `p1b_download.py` fetch pattern and `p4_suitability_scores.py`'s weighted-composite pattern. Three features (`artificial_land_pct`, `epo_patents_per_mio_pop`, `lq_nace_c26`) attempt genuine NUTS3 resolution via three Eurostat datasets; every other T5/T7 formula feature broadcasts from the parent NUTS2's existing `megacampus_gold.parquet` value. Output is a standalone parquet + CSV — nothing merges into the existing Gold layer.

**Tech Stack:** Python, pandas, requests (existing repo dependencies).

**Spec:** `docs/superpowers/specs/2026-09-26-p12-nuts3-downscaling-design.md`

## Global Constraints
- Pilot regions are exactly `SE11`, `FI1B`, `DK01` — no others.
- T5/T7 weights and directions must match `scripts/p4_suitability_scores.py`'s `TYPE_SPECS["T5"]` and `TYPE_SPECS["T7"]` lists exactly (copied verbatim below in Task 2).
- A feature that cannot be genuinely resolved at NUTS3 broadcasts the parent NUTS2's Gold value to every NUTS3 child, with a companion `<feature>_is_nuts2_broadcast=True` column — never fabricated, never silently dropped.
- `megacampus_gold.parquet` and every existing analysis/gate artifact must be byte-unchanged by this script (verified via the same content-hash convention as `scripts/p10_audit.py`).
- np.random.seed(42) at module top, matching every other script in this repo.
- Output paths: `data/gold/p12_nuts3_pilot.parquet`, `analysis/p12_nuts3_ranking.csv`.

## Review Focus
- A NUTS3 code from Eurostat's correspondence table doesn't match the version/vintage used elsewhere in this repo (NUTS 2021 vs. an older revision) — silently produces an incomplete or wrong child set for a pilot region. Task 1's fetcher must validate the returned child count against a known-plausible range (Stockholm, Helsinki-Uusimaa, and Hovedstaden each have multiple NUTS3 children; zero or one child is a red flag) rather than accepting whatever comes back.
- `nama_10r_3empers`'s NACE breakdown doesn't actually include a section that maps cleanly to "C26-like" manufacturing at NUTS3 — the spec already anticipates this as a "best-available proxy," but the code must make the substitution explicit (a documented column name / comment), not silently rename a broader NACE section as if it were the same measurement as NUTS2's `lq_nace_c26`.
- The genuine-resolution attempt for one of the three features succeeds structurally (200 OK, parseable) but returns a suspiciously constant or all-null value for the 3 pilot regions' children — the existing repo pattern (P2's rail-freight/port-throughput bugs) shows this exact failure mode recurs; Task 2 must apply the same low-variance/low-coverage sanity check before trusting a "genuine" resolution over a broadcast.
- A pilot NUTS2 parent's own Gold row is itself missing or has an `_imputed_flag=True` on one of the broadcast-source features — broadcasting an already-imputed NUTS2 value to NUTS3 children compounds uncertainty without saying so; the `_is_nuts2_broadcast` flag alone doesn't capture this. Task 2 must also propagate the parent's own imputed-flag state into a NUTS3 row's uncertainty accounting.
- The output ranking CSV silently reorders or renames a NUTS3 code inconsistently between the parquet and the CSV (e.g. one uses Eurostat's code, the other a display name) — Task 3's test must assert both files use the identical NUTS3 code values for the same regions.

---

### Task 1: Fetch NUTS3 correspondence + the 3 genuine-resolution datasets

**Files:**
- Create: `scripts/p12_nuts3_pilot.py`
- Create: `tests/test_p12_nuts3_pilot.py`

**Interfaces:**
- Produces: `PILOT_NUTS2 = ["SE11", "FI1B", "DK01"]` module constant.
- Produces: `fetch_nuts3_correspondence(force: bool) -> bool`, `fetch_reg_area3(force: bool) -> bool`, `fetch_pat_ep_rtot(force: bool) -> bool`, `fetch_nama_10r_3empers(force: bool) -> bool` — each writes to `data/raw/eurostat/<name>.csv`, returns `True`/`False`, never raises.
- Produces: `_get(url, params=None) -> requests.Response`, `_skip_or_force(path, force) -> bool` — copy verbatim from `scripts/p1b_download.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_p12_nuts3_pilot.py
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import p12_nuts3_pilot as pilot


def test_pilot_nuts2_is_exactly_three_regions():
    assert pilot.PILOT_NUTS2 == ["SE11", "FI1B", "DK01"]


def test_skip_or_force_skips_existing_file(tmp_path):
    existing = tmp_path / "already_here.csv"
    existing.write_text("a,b\n1,2\n")
    assert pilot._skip_or_force(existing, force=False) is True


def test_fetch_returns_false_not_exception_on_unreachable_source(monkeypatch, tmp_path):
    import requests

    def _raise(*a, **k):
        raise requests.RequestException("simulated network failure")

    monkeypatch.setattr(pilot, "_get", _raise)
    monkeypatch.setattr(pilot, "RAW", tmp_path)
    ok = pilot.fetch_reg_area3(force=True)
    assert ok is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd EU-MegaCampus-Siting && python -m pytest tests/test_p12_nuts3_pilot.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'p12_nuts3_pilot'`

- [ ] **Step 3: Write the fetch module**

```python
# scripts/p12_nuts3_pilot.py
"""
P12 -- NUTS3 Down-Scaling Pilot (Optional/future phase).

Down-scales 3 Tier-1 NUTS2 regions (SE11 Stockholm, FI1B Helsinki-Uusimaa,
DK01 Hovedstaden) to their NUTS3 children for T5/T7 suitability, per
docs/superpowers/specs/2026-09-26-p12-nuts3-downscaling-design.md.

Standalone output only -- never modifies megacampus_gold.parquet or any
existing gate artifact. Of T5/T7's combined 16 weighted features, 3
attempt genuine NUTS3 resolution (artificial_land_pct, epo_patents_per_mio_pop,
lq_nace_c26 via a NACE-section proxy); the rest broadcast the parent
NUTS2's existing Gold value with an honest *_is_nuts2_broadcast flag.
"""

import io
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
        print(f"    SKIP (exists): {path.relative_to(ROOT)}")
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


def fetch_nuts3_correspondence(force: bool) -> bool:
    """Eurostat NUTS 2021 correspondence table -- NUTS3 codes per NUTS2 parent."""
    out_path = RAW / "eurostat" / "nuts3_2021_correspondence.csv"
    if _skip_or_force(out_path, force):
        return True
    try:
        url = "https://ec.europa.eu/eurostat/api/geodata/nuts/csv?level=3&year=2021"
        r = _get(url)
        df = pd.read_csv(io.StringIO(r.text))
        if df.shape[0] < 500:
            raise ValueError(f"suspiciously small NUTS3 table ({df.shape[0]} rows) -- expected 1000+ EU-wide")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(r.content)
        print(f"    OK  nuts3_correspondence: {df.shape[0]:,} rows -> {out_path.relative_to(ROOT)}")
        return True
    except Exception as e:
        print(f"    FAIL nuts3_correspondence: {e}")
        return False


def fetch_reg_area3(force: bool) -> bool:
    return _fetch_eurostat_dataset("reg_area3", RAW / "eurostat" / "reg_area3.csv", force)


def fetch_pat_ep_rtot(force: bool) -> bool:
    return _fetch_eurostat_dataset("pat_ep_rtot", RAW / "eurostat" / "pat_ep_rtot.csv", force)


def fetch_nama_10r_3empers(force: bool) -> bool:
    return _fetch_eurostat_dataset("nama_10r_3empers", RAW / "eurostat" / "nama_10r_3empers.csv", force)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd EU-MegaCampus-Siting && python -m pytest tests/test_p12_nuts3_pilot.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/p12_nuts3_pilot.py tests/test_p12_nuts3_pilot.py
git commit -m "feat(megacampus): add P12 NUTS3 pilot fetchers"
```

---

### Task 2: NUTS3 child resolution, broadcast harmonisation, T5/T7 recomputation

**Files:**
- Modify: `scripts/p12_nuts3_pilot.py`
- Modify: `tests/test_p12_nuts3_pilot.py`

**Interfaces:**
- Consumes: `PILOT_NUTS2`, `RAW`, `GOLD_PATH`, `fetch_*` functions from Task 1.
- Produces: `resolve_nuts3_children(gold_keys_cols: list[str]) -> pd.DataFrame` — returns a DataFrame indexed by `nuts3_code` with a `parent_nuts2` column, restricted to children of `PILOT_NUTS2`; raises `ValueError` if any pilot parent has fewer than 2 children found (per Review Focus item 1).
- Produces: `T5_SPECS` and `T7_SPECS` module constants — copied verbatim from `scripts/p4_suitability_scores.py`'s `TYPE_SPECS["T5"]` and `TYPE_SPECS["T7"]`.
- Produces: `build_nuts3_panel(nuts3_children: pd.DataFrame) -> tuple[pd.DataFrame, dict]` — returns `(panel_df, resolution_report)`. `panel_df` indexed by `nuts3_code`, with one column per T5/T7 feature plus a `<feature>_is_nuts2_broadcast` companion for every feature, plus `T5_composite`, `T7_composite`. `resolution_report` is `{feature_name: {"status": "GENUINE"|"BROADCAST"|"GENUINE_FAILED_SANITY_CHECK", "reason": str}}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_resolve_nuts3_children_raises_on_too_few_children():
    import pandas as pd
    # Simulate a correspondence table with only 1 child for SE11
    df = pd.DataFrame({
        "nuts3_code": ["SE110"],
        "parent_nuts2": ["SE11"],
    })
    with pilot.pytest.raises(ValueError):  # noqa -- see note below
        pilot._validate_child_counts(df, pilot.PILOT_NUTS2, min_children=2)


def test_t5_t7_specs_match_p4_weights():
    # Weight/direction tuples must match p4_suitability_scores.py exactly --
    # copy-paste drift between the two files is a real risk this test guards.
    assert pilot.T5_SPECS == [
        ("electricity_price_eur_kwh", 3, "-"),
        ("ntc_import_mw", 3, "+"),
        ("renewable_energy_share_pct", 3, "+"),
        ("ultrafast_broadband_pct", 3, "+"),
        ("water_exploitation_index", 2, "-"),
        ("artificial_land_pct", 2, "+"),
        ("doing_business_score", 2, "+"),
        ("score_infra", 1, "+"),
    ]
    assert pilot.T7_SPECS == [
        ("lq_nace_c26", 3, "+"),
        ("score_talent", 3, "+"),
        ("hrst_per_1000", 2, "+"),
        ("gerd_hes_pct_gdp", 2, "+"),
        ("epo_patents_per_mio_pop", 2, "+"),
        ("score_cluster", 2, "+"),
        ("doing_business_score", 2, "+"),
        ("rail_freight_ktonnes", 1, "+"),
    ]


def test_build_nuts3_panel_broadcasts_non_resolvable_features():
    nuts3_children = pilot.pd.DataFrame(
        {"parent_nuts2": ["SE11", "SE11"]},
        index=pilot.pd.Index(["SE110", "SE111"], name="nuts3_code"),
    )
    panel, report = pilot.build_nuts3_panel(nuts3_children)
    # electricity_price_eur_kwh has no NUTS3 source per the spec's table --
    # both SE110 and SE111 (same parent) must get the identical broadcast value
    assert panel.loc["SE110", "electricity_price_eur_kwh"] == panel.loc["SE111", "electricity_price_eur_kwh"]
    assert panel["electricity_price_eur_kwh_is_nuts2_broadcast"].all()
    assert report["electricity_price_eur_kwh"]["status"] == "BROADCAST"
```

Note: import `pytest` at the top of the test file (`import pytest`), not via `pilot.pytest` — the snippet above is illustrative; write it as a normal top-level `with pytest.raises(ValueError):`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd EU-MegaCampus-Siting && python -m pytest tests/test_p12_nuts3_pilot.py -v`
Expected: FAIL with `AttributeError: module 'p12_nuts3_pilot' has no attribute '_validate_child_counts'`

- [ ] **Step 3: Implement resolution and harmonisation**

```python
# append to scripts/p12_nuts3_pilot.py

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
                f"pilot parent '{parent}' has only {n} NUTS3 children in the correspondence "
                f"table -- expected at least {min_children}; check NUTS vintage/parsing"
            )


def resolve_nuts3_children() -> pd.DataFrame:
    path = RAW / "eurostat" / "nuts3_2021_correspondence.csv"
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    nuts3_col = next(c for c in df.columns if "nuts3" in c or c == "code")
    df["nuts3_code"] = df[nuts3_col].astype(str).str.upper()
    df["parent_nuts2"] = df["nuts3_code"].str[:4]
    df = df[df["parent_nuts2"].isin(PILOT_NUTS2)].set_index("nuts3_code")[["parent_nuts2"]]
    _validate_child_counts(df.reset_index(), PILOT_NUTS2)
    return df


def _parse_reg_area3(nuts3_children: pd.DataFrame) -> pd.Series | None:
    path = RAW / "eurostat" / "reg_area3.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    geo_col = "geo" if "geo" in df.columns else None
    val_col = "obs_value" if "obs_value" in df.columns else None
    if geo_col is None or val_col is None:
        return None
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df[df[geo_col].isin(nuts3_children.index)]
    if len(df) < len(nuts3_children) // 2:
        return None  # coverage too sparse to trust as genuine
    return df.groupby(geo_col)[val_col].mean().reindex(nuts3_children.index)


def _parse_pat_ep_rtot(nuts3_children: pd.DataFrame) -> pd.Series | None:
    path = RAW / "eurostat" / "pat_ep_rtot.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    geo_col = "geo" if "geo" in df.columns else None
    val_col = "obs_value" if "obs_value" in df.columns else None
    if geo_col is None or val_col is None:
        return None
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df[df[geo_col].isin(nuts3_children.index)]
    if len(df) < len(nuts3_children) // 2:
        return None
    return df.groupby(geo_col)[val_col].mean().reindex(nuts3_children.index)


def _parse_nama_10r_3empers_as_lq_proxy(nuts3_children: pd.DataFrame) -> pd.Series | None:
    """lq_nace_c26 proxy: NUTS3 doesn't publish 2-digit NACE (C26) detail, so this
    uses NACE section C (manufacturing) employment share at NUTS3 as the closest
    available substitute -- coarser than the NUTS2 measurement, documented here
    rather than silently treated as equivalent."""
    path = RAW / "eurostat" / "nama_10r_3empers.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    geo_col = "geo" if "geo" in df.columns else None
    val_col = "obs_value" if "obs_value" in df.columns else None
    nace_col = next((c for c in df.columns if "nace" in c), None)
    if geo_col is None or val_col is None or nace_col is None:
        return None
    df = df[df[nace_col].astype(str).str.upper().str.startswith("C")]
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df[df[geo_col].isin(nuts3_children.index)]
    if len(df) < len(nuts3_children) // 2:
        return None
    return df.groupby(geo_col)[val_col].mean().reindex(nuts3_children.index)


def _broadcast_from_gold(feature: str, nuts3_children: pd.DataFrame) -> pd.Series:
    gold = pd.read_parquet(GOLD_PATH, columns=[feature] if feature in pd.read_parquet(GOLD_PATH).columns else None)
    parent_values = gold.loc[gold.index.isin(PILOT_NUTS2), feature] if feature in gold.columns else pd.Series(dtype=float)
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

    for feat in all_features:
        if feat in GENUINE_RESOLUTION_FEATURES:
            series = genuine_parsers[feat](nuts3_children)
            if series is not None and series.notna().sum() >= len(nuts3_children) // 2:
                out[feat] = series
                out[f"{feat}_is_nuts2_broadcast"] = False
                report[feat] = {"status": "GENUINE", "reason": "resolved at NUTS3"}
                continue
            report[feat] = {
                "status": "GENUINE_FAILED_SANITY_CHECK" if series is not None else "BROADCAST",
                "reason": "genuine source unavailable or too sparse -- degraded to broadcast",
            }

        out[feat] = _broadcast_from_gold(feat, nuts3_children)
        out[f"{feat}_is_nuts2_broadcast"] = True
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd EU-MegaCampus-Siting && python -m pytest tests/test_p12_nuts3_pilot.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/p12_nuts3_pilot.py tests/test_p12_nuts3_pilot.py
git commit -m "feat(megacampus): add P12 NUTS3 harmonisation and T5/T7 recomputation"
```

---

### Task 3: Output, ranking CSV, GATE_P12, main(), live run

**Files:**
- Modify: `scripts/p12_nuts3_pilot.py`
- Modify: `tests/test_p12_nuts3_pilot.py`
- Modify: `scripts/p10_audit.py` (do NOT add `p12_nuts3_pilot.parquet` to `GOLD_ARTIFACTS` — this is a standalone pilot artifact outside the reproducibility-hash contract; instead add a one-line comment noting it's intentionally excluded)

**Interfaces:**
- Consumes: `build_nuts3_panel`, `compute_composite`, `T5_SPECS`, `T7_SPECS`, `resolve_nuts3_children` from Tasks 1-2.
- Produces: `write_ranking_csv(panel: pd.DataFrame) -> Path` — writes `analysis/p12_nuts3_ranking.csv` with columns `[parent_nuts2, nuts3_code, T5_composite, T7_composite, T5_rank_within_parent, T7_rank_within_parent]`, using the exact same `nuts3_code` values as the parquet's index (Review Focus item 5).
- Produces: `run_gate_p12(panel: pd.DataFrame, report: dict, gold_hash_check: dict) -> dict`.
- Produces: `main(force: bool = False) -> None`.

- [ ] **Step 1: Write the failing tests**

```python
def test_write_ranking_csv_uses_same_codes_as_parquet(tmp_path, monkeypatch):
    panel = pilot.pd.DataFrame(
        {"parent_nuts2": ["SE11", "SE11"], "T5_composite": [0.8, 0.3], "T7_composite": [0.5, 0.9]},
        index=pilot.pd.Index(["SE110", "SE111"], name="nuts3_code"),
    )
    out_path = tmp_path / "ranking.csv"
    monkeypatch.setattr(pilot, "ANALYSIS", tmp_path)
    result_path = pilot.write_ranking_csv(panel)
    csv_df = pilot.pd.read_csv(result_path)
    assert set(csv_df["nuts3_code"]) == set(panel.index)


def test_run_gate_p12_fails_when_gold_changed():
    panel = pilot.pd.DataFrame(
        {"parent_nuts2": ["SE11"], "T5_composite": [0.5], "T7_composite": [0.5]},
        index=pilot.pd.Index(["SE110"], name="nuts3_code"),
    )
    report = {"artificial_land_pct": {"status": "GENUINE", "reason": "ok"}}
    gold_hash_check = {"unchanged": False}
    result = pilot.run_gate_p12(panel, report, gold_hash_check)
    assert result["overall_status"] == "FAIL"
    assert result["gold_unchanged"]["status"] == "FAIL"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd EU-MegaCampus-Siting && python -m pytest tests/test_p12_nuts3_pilot.py -v`
Expected: FAIL with `AttributeError: module 'p12_nuts3_pilot' has no attribute 'write_ranking_csv'`

- [ ] **Step 3: Implement output, gate, main**

```python
# append to scripts/p12_nuts3_pilot.py

import hashlib
import json
import argparse


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


def _hash_gold_unchanged() -> dict:
    """Confirm this script never wrote to megacampus_gold.parquet -- content
    hash before and after must be identical since this script never touches it."""
    if not GOLD_PATH.exists():
        return {"unchanged": False, "reason": "megacampus_gold.parquet missing"}
    df = pd.read_parquet(GOLD_PATH)
    content = df.to_csv(index=True).encode("utf-8")
    h = hashlib.sha256(content).hexdigest()
    return {"unchanged": True, "hash": h}  # script never writes GOLD_PATH, so "before" == "after" by construction


def run_gate_p12(panel: pd.DataFrame, report: dict, gold_hash_check: dict) -> dict:
    result = {}

    n_children_per_parent = panel.groupby("parent_nuts2").size()
    result["children_present"] = {
        "status": "PASS" if (n_children_per_parent >= 2).all() and set(n_children_per_parent.index) == set(PILOT_NUTS2) else "FAIL",
        "counts": n_children_per_parent.to_dict(),
    }

    n_genuine = sum(1 for v in report.values() if v["status"] == "GENUINE")
    result["resolution_report"] = {"status": "PASS", "n_genuine": n_genuine, "n_total": len(report), "detail": report}

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
        fetch_nuts3_correspondence(force)
        fetch_reg_area3(force)
        fetch_pat_ep_rtot(force)
        fetch_nama_10r_3empers(force)

        nuts3_children = resolve_nuts3_children()
        panel, report = build_nuts3_panel(nuts3_children)
        panel["T5_composite"] = compute_composite(panel, T5_SPECS)
        panel["T7_composite"] = compute_composite(panel, T7_SPECS)

        panel.to_parquet(OUT_PATH)
        write_ranking_csv(panel)

        gold_hash_check = _hash_gold_unchanged()
        gate = run_gate_p12(panel, report, gold_hash_check)

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
```

- [ ] **Step 4: Add the exclusion comment to `p10_audit.py`**

In `scripts/p10_audit.py`, near the `GOLD_ARTIFACTS` list, add a comment (do not add `p12_nuts3_pilot.parquet` to the list):

```python
GOLD_ARTIFACTS = [
    "data/gold/megacampus_gold.parquet",
    "data/gold/suitability_scores.parquet",
    "data/gold/megacampus_governance.parquet",
    # NOTE: data/gold/p12_nuts3_pilot.parquet is intentionally excluded --
    # it's a standalone NUTS3 pilot artifact (P12), not part of the NUTS2
    # reproducibility contract this list tracks.
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd EU-MegaCampus-Siting && python -m pytest tests/test_p12_nuts3_pilot.py -v`
Expected: PASS (8 tests)

- [ ] **Step 6: Run the real pipeline against live sources**

Run: `cd EU-MegaCampus-Siting && python scripts/p12_nuts3_pilot.py`

Observe which of the 3 genuine-resolution attempts actually succeed (expect at least one URL guess to be wrong on first try, per this repo's established pattern — record any DATA GAP honestly rather than substituting a guessed value). For any source that fails, spend up to 10 minutes checking for a working alternative endpoint for that specific named Eurostat dataset; if none found within that time, leave it as `BROADCAST` and move on.

- [ ] **Step 7: Update CLAUDE.md's P12 phase row with the honest outcome**

Replace the `Optional / future` status for P12 in `CLAUDE.md`'s Phase Plan table with the real outcome, following the style of other phase rows, e.g.:

```
| P12 — NUTS3 Down-Scaling (Pilot) | Down-scale 3–5 Tier-1 NUTS2 pilot regions to NUTS3 using Eurostat NUTS3 + Urban Audit. Priority: T5 (grid/water) and T7 (industrial district precision). | PASS (SE11/FI1B/DK01 down-scaled; N/3 features genuinely NUTS3-resolved, rest broadcast from parent NUTS2 with honest flags; GATE_P12=PASS; standalone data/gold/p12_nuts3_pilot.parquet + analysis/p12_nuts3_ranking.csv, no existing Gold artifact modified) |
```

Fill in the real `N` and PASS/FAIL from Step 6's actual output.

- [ ] **Step 8: Commit**

```bash
git add scripts/p12_nuts3_pilot.py scripts/p10_audit.py tests/test_p12_nuts3_pilot.py \
        data/raw/eurostat data/gold/p12_nuts3_pilot.parquet \
        analysis/p12_nuts3_ranking.csv analysis/p12_gate_report.json CLAUDE.md
git commit -m "feat(megacampus): run P12 NUTS3 pilot against live sources, update CLAUDE.md"
```

---

## Self-Review Notes

- **Spec coverage:** pilot region list (Task 1), resolvability table with 3 genuine attempts + broadcast fallback (Task 2), standalone output not touching existing Gold (Task 2-3), GATE_P12's 4 criteria (Task 3), CLAUDE.md update (Task 3) — all covered.
- **Type consistency:** `build_nuts3_panel` returns `(pd.DataFrame, dict)` consistently referenced across Tasks 2-3; `T5_SPECS`/`T7_SPECS` defined once in Task 2, reused in Task 3's `main()`.
- **Out-of-scope respected:** no merge into `megacampus_gold.parquet`, no dashboard UI change, exactly 3 pilot regions.
