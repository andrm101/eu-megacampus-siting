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
