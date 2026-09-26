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
