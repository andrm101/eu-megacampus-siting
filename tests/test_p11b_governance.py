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
        {"rda_capacity_index": [3, 5]},
        index=gov.pd.Index(["FR10", "DE30"], name="nuts2_code"),
    )
    try:
        gov.append_to_gold(governance_df)
        assert False, "expected ValueError on column collision"
    except ValueError as e:
        assert "rda_capacity_index" in str(e)

    reloaded = gov.pd.read_parquet(gold_path)
    assert reloaded.loc["FR10", "rda_capacity_index"] == 10


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
