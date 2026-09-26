import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import p11b_governance as gov


def test_skip_or_force_skips_existing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(gov, "ROOT", tmp_path)
    existing = tmp_path / "already_here.csv"
    existing.write_text("a,b\n1,2\n")
    assert gov._skip_or_force(existing, force=False) is True


def test_skip_or_force_does_not_skip_with_force(tmp_path, monkeypatch):
    monkeypatch.setattr(gov, "ROOT", tmp_path)
    existing = tmp_path / "already_here.csv"
    existing.write_text("a,b\n1,2\n")
    assert gov._skip_or_force(existing, force=True) is False


def test_fetch_returns_false_not_exception_on_unreachable_source(monkeypatch, tmp_path):
    import requests

    def _raise(*a, **k):
        raise requests.RequestException("simulated network failure")

    monkeypatch.setattr(gov, "_get", _raise)
    monkeypatch.setattr(gov, "RAW", tmp_path)
    ok = gov.fetch_esif_programmes(force=True)
    assert ok is False


def test_broadcast_country_to_nuts2_maps_correctly():
    gold_keys = gov.pd.DataFrame(
        {"country_code": ["FR", "FR", "DE"]},
        index=gov.pd.Index(["FR10", "FR20", "DE30"], name="nuts2_code"),
    )
    country_values = gov.pd.Series({"FR": 42.0, "DE": 55.0})
    result = gov.broadcast_country_to_nuts2(country_values, gold_keys)
    assert result["FR10"] == 42.0
    assert result["FR20"] == 42.0
    assert result["DE30"] == 55.0


def test_broadcast_country_to_nuts2_nulls_unmatched_country():
    gold_keys = gov.pd.DataFrame(
        {"country_code": ["XX"]},
        index=gov.pd.Index(["XX10"], name="nuts2_code"),
    )
    country_values = gov.pd.Series({"FR": 42.0})
    result = gov.broadcast_country_to_nuts2(country_values, gold_keys)
    assert gov.pd.isna(result["XX10"])


def test_harmonise_governance_records_data_gap_when_all_fetches_fail():
    gold_keys = gov.pd.DataFrame(
        {"country_code": ["FR", "DE"]},
        index=gov.pd.Index(["FR10", "DE30"], name="nuts2_code"),
    )
    fetch_results = {
        "s3_strategies": False, "cluster_registry": False,
        "educ_institutions": False, "esif_programmes": False,
        "oecd_fiscal_autonomy": False,
    }
    governance_df, gap_report = gov.harmonise_governance(gold_keys, fetch_results)
    assert len(governance_df) == 2
    assert governance_df["rda_capacity_index"].isna().all()
    assert governance_df["rda_capacity_index_imputed_flag"].all()
    assert gap_report["rda_capacity_index"]["status"] == "DATA_GAP"


def test_append_to_gold_preserves_pre_existing_columns(tmp_path, monkeypatch):
    gold_df = gov.pd.DataFrame(
        {"country_code": ["FR", "DE"], "population": [1000, 2000]},
        index=gov.pd.Index(["FR10", "DE30"], name="nuts2_code"),
    )
    gold_path = tmp_path / "gold.parquet"
    gov_gold_path = tmp_path / "governance.parquet"
    gold_df.to_parquet(gold_path)
    monkeypatch.setattr(gov, "GOLD_PATH", gold_path)
    monkeypatch.setattr(gov, "GOVERNANCE_GOLD_PATH", gov_gold_path)

    governance_df = gov.pd.DataFrame(
        {"rda_capacity_index": [3, 5]},
        index=gov.pd.Index(["FR10", "DE30"], name="nuts2_code"),
    )
    result = gov.append_to_gold(governance_df)
    assert result["unchanged"] is True

    reloaded = gov.pd.read_parquet(gold_path)
    assert reloaded.loc["FR10", "population"] == 1000
    assert reloaded.loc["DE30", "rda_capacity_index"] == 5


def test_append_to_gold_raises_on_column_collision(tmp_path, monkeypatch):
    # Use a column name that is NOT one of the governance script's own
    # outputs -- a genuine, unrelated pre-existing Gold column should still
    # block the run. ("legacy_unrelated_column" is not in GOVERNANCE_COLUMNS.)
    gold_df = gov.pd.DataFrame(
        {"country_code": ["FR", "DE"], "legacy_unrelated_column": [10, 20]},
        index=gov.pd.Index(["FR10", "DE30"], name="nuts2_code"),
    )
    gold_path = tmp_path / "gold.parquet"
    gov_gold_path = tmp_path / "governance.parquet"
    gold_df.to_parquet(gold_path)
    monkeypatch.setattr(gov, "GOLD_PATH", gold_path)
    monkeypatch.setattr(gov, "GOVERNANCE_GOLD_PATH", gov_gold_path)

    governance_df = gov.pd.DataFrame(
        {"legacy_unrelated_column": [3, 5]},
        index=gov.pd.Index(["FR10", "DE30"], name="nuts2_code"),
    )
    try:
        gov.append_to_gold(governance_df)
        assert False, "expected ValueError on column collision"
    except ValueError as e:
        assert "legacy_unrelated_column" in str(e)

    reloaded = gov.pd.read_parquet(gold_path)
    assert reloaded.loc["FR10", "legacy_unrelated_column"] == 10


def test_append_to_gold_does_not_raise_when_own_governance_columns_already_present(tmp_path, monkeypatch):
    # Simulate a second run: Gold already has this script's own output from
    # a prior run. This must NOT raise -- it should overwrite with fresh
    # values (idempotent re-run), per the I1 fix.
    gold_df = gov.pd.DataFrame(
        {"country_code": ["FR", "DE"], "rda_capacity_index": [10, 20]},
        index=gov.pd.Index(["FR10", "DE30"], name="nuts2_code"),
    )
    gold_path = tmp_path / "gold.parquet"
    gov_gold_path = tmp_path / "governance.parquet"
    gold_df.to_parquet(gold_path)
    monkeypatch.setattr(gov, "GOLD_PATH", gold_path)
    monkeypatch.setattr(gov, "GOVERNANCE_GOLD_PATH", gov_gold_path)

    governance_df = gov.pd.DataFrame(
        {"rda_capacity_index": [99, 55]},
        index=gov.pd.Index(["FR10", "DE30"], name="nuts2_code"),
    )
    result = gov.append_to_gold(governance_df)
    assert result is not None

    reloaded = gov.pd.read_parquet(gold_path)
    assert reloaded.loc["FR10", "rda_capacity_index"] == 99
    assert reloaded.loc["DE30", "rda_capacity_index"] == 55


def test_parse_educ_institutions_signals_broadcast_on_sparse_nuts2_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(gov, "RAW", tmp_path)
    out_path = tmp_path / "eurostat" / "educ_uoe_enrt01.csv"
    out_path.parent.mkdir(parents=True)
    # Only a handful of NUTS2-coded rows (< 20) -- should degrade to broadcast.
    raw = gov.pd.DataFrame({
        "sex": ["T"] * 5,
        "sector": ["TOT_SEC"] * 5,
        "isced11": ["ED5-8"] * 5,
        "geo": ["FR10", "FR20", "DE30", "FR10", "DE30"],
        "obs_value": [100, 200, 300, 110, 310],
    })
    raw.to_csv(out_path, index=False)

    gold_keys = gov.pd.DataFrame(
        {"country_code": ["FR", "FR", "DE"]},
        index=gov.pd.Index(["FR10", "FR20", "DE30"], name="nuts2_code"),
    )
    series, is_broadcast = gov._parse_educ_institutions(gold_keys)
    assert is_broadcast is True
    assert series is not None
    assert not series.isna().all()


def test_parse_educ_institutions_no_broadcast_when_nuts2_coverage_sufficient(tmp_path, monkeypatch):
    monkeypatch.setattr(gov, "RAW", tmp_path)
    out_path = tmp_path / "eurostat" / "educ_uoe_enrt01.csv"
    out_path.parent.mkdir(parents=True)
    nuts2_codes = [f"FR{i:02d}" for i in range(25)]
    raw = gov.pd.DataFrame({
        "sex": ["T"] * 25,
        "sector": ["TOT_SEC"] * 25,
        "isced11": ["ED5-8"] * 25,
        "geo": nuts2_codes,
        "obs_value": list(range(25)),
    })
    raw.to_csv(out_path, index=False)

    gold_keys = gov.pd.DataFrame(
        {"country_code": ["FR"] * 25},
        index=gov.pd.Index(nuts2_codes, name="nuts2_code"),
    )
    series, is_broadcast = gov._parse_educ_institutions(gold_keys)
    assert is_broadcast is False
    assert series is not None


def test_harmonise_governance_flags_educ_institutions_as_country_broadcast(tmp_path, monkeypatch):
    monkeypatch.setattr(gov, "RAW", tmp_path)
    out_path = tmp_path / "eurostat" / "educ_uoe_enrt01.csv"
    out_path.parent.mkdir(parents=True)
    raw = gov.pd.DataFrame({
        "sex": ["T"] * 3,
        "sector": ["TOT_SEC"] * 3,
        "isced11": ["ED5-8"] * 3,
        "geo": ["FR10", "FR20", "DE30"],
        "obs_value": [100, 200, 300],
    })
    raw.to_csv(out_path, index=False)

    gold_keys = gov.pd.DataFrame(
        {"country_code": ["FR", "FR", "DE"]},
        index=gov.pd.Index(["FR10", "FR20", "DE30"], name="nuts2_code"),
    )
    fetch_results = {
        "s3_strategies": False, "cluster_registry": False,
        "educ_institutions": True, "esif_programmes": False,
        "oecd_fiscal_autonomy": False,
    }
    governance_df, gap_report = gov.harmonise_governance(gold_keys, fetch_results)
    assert governance_df["institutional_diversity_score_imputed_flag"].all()
    assert gap_report["institutional_diversity_score"]["status"] == "COUNTRY_BROADCAST"


def test_run_gate_p11b_fails_on_wrong_row_count():
    governance_df = gov.pd.DataFrame(
        {"rda_capacity_index": [1]},
        index=gov.pd.Index(["FR10"], name="nuts2_code"),
    )
    gap_report = {"rda_capacity_index": {"status": "OK", "n_null": 0}}
    hash_check = {"unchanged": True}
    result = gov.run_gate_p11b(governance_df, gap_report, hash_check, expected_n_regions=242)
    assert result["overall_status"] == "FAIL"
    assert "row_count" in result
