import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import p12_nuts3_pilot as pilot
import pandas as pd


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


def test_validate_child_counts_raises_on_too_few_children():
    df = pd.DataFrame({
        "nuts3_code": ["SE110"],
        "parent_nuts2": ["SE11"],
    })
    with pytest.raises(ValueError):
        pilot._validate_child_counts(df, pilot.PILOT_NUTS2, min_children=2)


def test_t5_t7_specs_match_p4_weights():
    # Import the LIVE specs from p4_suitability_scores.py -- a hard-coded
    # literal here would pass even after p4's weights drifted (this exact
    # gap was found in final review: the previous version of this test
    # never imported p4 at all).
    import p4_suitability_scores as p4
    assert pilot.T5_SPECS == p4.TYPE_SPECS["T5"]
    assert pilot.T7_SPECS == p4.TYPE_SPECS["T7"]


def test_parse_pat_ep_rtot_filters_to_per_capita_unit_and_latest_year(tmp_path, monkeypatch):
    # Real bug found in final review: the original parser averaged obs_value
    # across ALL unit dimensions (NR counts, P_MHAB per-capita, GDP_BEUR,
    # P_MACT), producing a meaningless blended magnitude. Must filter to
    # unit=='P_MHAB' (patents per million inhabitants) and use the latest year.
    eurostat_dir = tmp_path / "eurostat"
    eurostat_dir.mkdir()
    df = pd.DataFrame({
        "geo": ["SE110", "SE110", "SE110", "SE110"],
        "unit": ["NR", "P_MHAB", "P_MHAB", "GDP_BEUR"],
        "time_period": [2012, 2011, 2012, 2012],
        "obs_value": [50000.0, 400.0, 396.849, 12.5],
    })
    df.to_csv(eurostat_dir / "pat_ep_rtot.csv", index=False)
    monkeypatch.setattr(pilot, "RAW", tmp_path)

    nuts3_children = pd.DataFrame(
        {"parent_nuts2": ["SE11"]},
        index=pd.Index(["SE110"], name="nuts3_code"),
    )
    result = pilot._parse_pat_ep_rtot(nuts3_children)
    # Must pick the P_MHAB, most-recent-year (2012) value -- 396.849 -- not
    # a mean across NR/GDP_BEUR/mixed years.
    assert result["SE110"] == pytest.approx(396.849)


def test_parse_reg_area3_degrades_to_broadcast_no_real_artificial_land_source():
    # Real bug found in final review: reg_area3's landuse codes are only
    # L0008 (land area, i.e. total minus water) and TOTAL (including water)
    # -- the L0008/TOTAL ratio is ~93-98% for every region tested, which is
    # land-vs-water, not artificial/built-up land share. There is no genuine
    # artificial-land-percentage source in this dataset; it must degrade to
    # BROADCAST rather than reporting a nonsensical ~95% "artificial land".
    nuts3_children = pd.DataFrame(
        {"parent_nuts2": ["SE11"]},
        index=pd.Index(["SE110"], name="nuts3_code"),
    )
    assert pilot._parse_reg_area3(nuts3_children) is None


def test_parse_nama_10r_3empers_computes_real_location_quotient(tmp_path, monkeypatch):
    # Real bug found in final review: the original parser returned raw
    # manufacturing headcount in thousands (an absolute magnitude, not a
    # quotient). This test requires the correct LQ formula:
    # (region_C / region_TOTAL) / (country_C / country_TOTAL).
    eurostat_dir = tmp_path / "eurostat"
    eurostat_dir.mkdir()
    df = pd.DataFrame({
        "geo": ["SE110", "SE110", "SE", "SE"],
        "wstatus": ["EMP", "EMP", "EMP", "EMP"],
        "nace_r2": ["C", "TOTAL", "C", "TOTAL"],
        "time_period": [2023, 2023, 2023, 2023],
        "obs_value": [10.0, 200.0, 500.0, 5000.0],
    })
    df.to_csv(eurostat_dir / "nama_10r_3empers.csv", index=False)
    monkeypatch.setattr(pilot, "RAW", tmp_path)

    nuts3_children = pd.DataFrame(
        {"parent_nuts2": ["SE11"]},
        index=pd.Index(["SE110"], name="nuts3_code"),
    )
    result = pilot._parse_nama_10r_3empers_as_lq_proxy(nuts3_children)
    # region share = 10/200 = 0.05; country share = 500/5000 = 0.10
    # LQ = 0.05 / 0.10 = 0.5
    assert result["SE110"] == pytest.approx(0.5)


def test_sanity_check_report_compares_genuine_values_to_parent_gold(monkeypatch):
    # Spec's Gate criterion 3 requires a documented consistency figure for
    # genuinely-resolved features vs. the parent NUTS2's existing Gold value
    # -- final review found this was silently dropped from the shipped code.
    panel = pd.DataFrame(
        {
            "parent_nuts2": ["SE11"],
            "epo_patents_per_mio_pop": [400.0],
            "epo_patents_per_mio_pop_is_nuts2_broadcast": [False],
        },
        index=pd.Index(["SE110"], name="nuts3_code"),
    )
    report = {"epo_patents_per_mio_pop": {"status": "GENUINE", "reason": "resolved at NUTS3"}}

    def fake_gold_value(feature, parent):
        return 396.8

    monkeypatch.setattr(pilot, "_get_parent_gold_value", fake_gold_value)
    sanity = pilot.compute_sanity_check(panel, report)
    assert "epo_patents_per_mio_pop" in sanity
    assert sanity["epo_patents_per_mio_pop"]["SE11"]["nuts3_mean"] == pytest.approx(400.0)
    assert sanity["epo_patents_per_mio_pop"]["SE11"]["parent_gold_value"] == pytest.approx(396.8)


def test_build_nuts3_panel_broadcasts_non_resolvable_features(monkeypatch):
    nuts3_children = pd.DataFrame(
        {"parent_nuts2": ["SE11", "SE11"]},
        index=pd.Index(["SE110", "SE111"], name="nuts3_code"),
    )

    def fake_broadcast(feature, children):
        return pd.Series(1.0, index=children.index)

    monkeypatch.setattr(pilot, "_broadcast_from_gold", fake_broadcast)
    panel, report = pilot.build_nuts3_panel(nuts3_children)
    # electricity_price_eur_kwh has no NUTS3 source per the spec's table --
    # both SE110 and SE111 (same parent) must get the identical broadcast value
    assert panel.loc["SE110", "electricity_price_eur_kwh"] == panel.loc["SE111", "electricity_price_eur_kwh"]
    assert panel["electricity_price_eur_kwh_is_nuts2_broadcast"].all()
    assert report["electricity_price_eur_kwh"]["status"] == "BROADCAST"


def test_write_ranking_csv_uses_same_codes_as_parquet(tmp_path, monkeypatch):
    panel = pd.DataFrame(
        {"parent_nuts2": ["SE11", "SE11"], "T5_composite": [0.8, 0.3], "T7_composite": [0.5, 0.9]},
        index=pd.Index(["SE110", "SE111"], name="nuts3_code"),
    )
    monkeypatch.setattr(pilot, "ANALYSIS", tmp_path)
    result_path = pilot.write_ranking_csv(panel)
    csv_df = pd.read_csv(result_path)
    assert set(csv_df["nuts3_code"]) == set(panel.index)


def test_run_gate_p12_fails_when_gold_changed():
    panel = pd.DataFrame(
        {"parent_nuts2": ["SE11"], "T5_composite": [0.5], "T7_composite": [0.5]},
        index=pd.Index(["SE110"], name="nuts3_code"),
    )
    report = {"artificial_land_pct": {"status": "GENUINE", "reason": "ok"}}
    gold_hash_check = {"unchanged": False}
    result = pilot.run_gate_p12(panel, report, gold_hash_check)
    assert result["overall_status"] == "FAIL"
    assert result["gold_unchanged"]["status"] == "FAIL"


def test_hash_gold_unchanged_actually_compares_against_baseline(monkeypatch, tmp_path):
    # Real bug found in final review: the original _hash_gold_unchanged()
    # always returned unchanged=True unconditionally -- it never compared
    # against anything, so the gate criterion was vacuous by construction.
    gold_path = tmp_path / "gold.parquet"
    pd.DataFrame({"x": [1, 2]}, index=pd.Index(["A", "B"], name="nuts2_code")).to_parquet(gold_path)
    monkeypatch.setattr(pilot, "GOLD_PATH", gold_path)

    baseline = pilot._compute_gold_hash()
    result_unchanged = pilot._hash_gold_unchanged(expected_hash=baseline)
    assert result_unchanged["unchanged"] is True

    result_changed = pilot._hash_gold_unchanged(expected_hash="deadbeef" * 8)
    assert result_changed["unchanged"] is False


def test_resolution_report_computes_weight_based_coverage():
    # Real bug found in final review: coverage was counted by feature count,
    # not by weight-points, and the spec's "out of 20" framing was wrong --
    # T5's weights actually sum to 19 and T7's to 17.
    report = {
        "artificial_land_pct": {"status": "BROADCAST", "reason": "x"},
        "epo_patents_per_mio_pop": {"status": "GENUINE", "reason": "x"},
        "lq_nace_c26": {"status": "PROXY", "reason": "x"},
    }
    for f, _, _ in pilot.T5_SPECS + pilot.T7_SPECS:
        report.setdefault(f, {"status": "BROADCAST", "reason": "x"})
    coverage = pilot._compute_weight_coverage(report)
    assert coverage["T5"]["total_weight"] == sum(w for _, w, _ in pilot.T5_SPECS)
    assert coverage["T7"]["total_weight"] == sum(w for _, w, _ in pilot.T7_SPECS)
    # epo_patents_per_mio_pop has weight 2 in T7 and is GENUINE
    assert coverage["T7"]["genuine_or_proxy_weight"] >= 2
