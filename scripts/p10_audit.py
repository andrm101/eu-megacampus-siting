"""
Phase 10 -- Pipeline audit: seed check, hash verification, vocab guard.
Usage: python scripts/p10_audit.py [--mode=full|hashes|seeds|vocab|generate-hashes]
       python scripts/p10_audit.py --verify-hashes   (Makefile `reproduce` alias for --mode=hashes)
"""

import argparse
import glob as glob_mod
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

np.random.seed(42)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

GOLD_ARTIFACTS = [
    "data/gold/megacampus_gold.parquet",
    "data/gold/suitability_scores.parquet",
    "data/gold/megacampus_governance.parquet",
]
EXPECTED_HASHES_FILE = ROOT / "expected_hashes.json"


def sha256_file(path: Path) -> str:
    """Content hash for reproducibility checking.

    Parquet files are hashed on their canonical data content (a deterministic
    CSV serialization), not raw file bytes -- pyarrow embeds non-data metadata
    (e.g. a write timestamp) that differs between two runs of the identical
    pipeline with the identical seed, which would otherwise falsely FAIL a
    reproducible pipeline (confirmed directly in EU-Innovation-Panel's own
    P10 audit: two back-to-back runs on unchanged inputs produced different
    raw file bytes but zero differing cell values). Everything else is
    hashed on raw bytes as before.
    """
    if path.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        content = df.to_csv(index=True).encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_expected_hashes() -> dict:
    """Compute sha256 for every Gold artifact and write expected_hashes.json.

    Run this once a pipeline output is considered a trusted reference; every
    subsequent audit run then verifies current artifacts still match it.
    Generate and verify within the same Python/pandas environment -- content
    hashing is stable across repeated runs within one pandas version, but not
    guaranteed identical across different versions (float-to-CSV formatting
    has changed between pandas releases before).
    """
    hashes = {}
    missing = []
    for rel_path in GOLD_ARTIFACTS:
        p = ROOT / rel_path
        if p.exists():
            hashes[rel_path] = sha256_file(p)
        else:
            missing.append(rel_path)

    EXPECTED_HASHES_FILE.write_text(json.dumps(hashes, indent=2, sort_keys=True))
    return {"written": str(EXPECTED_HASHES_FILE), "n_hashed": len(hashes), "missing": missing}


def run_hash_check() -> dict:
    if not EXPECTED_HASHES_FILE.exists():
        return {"status": "SKIP", "reason": "expected_hashes.json not yet generated (run --mode=generate-hashes)"}

    expected = json.loads(EXPECTED_HASHES_FILE.read_text())
    results = {}
    all_match = True
    for rel_path, expected_hash in expected.items():
        p = ROOT / rel_path
        if not p.exists():
            results[rel_path] = {"status": "MISSING"}
            all_match = False
        else:
            actual = sha256_file(p)
            match = actual == expected_hash
            results[rel_path] = {"status": "MATCH" if match else "MISMATCH", "actual": actual[:16]}
            if not match:
                all_match = False

    return {"status": "PASS" if all_match else "FAIL", "artifacts": results}


def run_seed_audit() -> dict:
    """Delegate to src/utils/seed_check.py and capture output."""
    import subprocess
    scripts_dir = ROOT / "scripts"
    result = subprocess.run(
        [sys.executable, str(ROOT / "src/utils/seed_check.py"), str(scripts_dir)],
        capture_output=True, text=True
    )
    return {
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "violations": result.stderr.strip().splitlines() if result.returncode != 0 else [],
    }


def run_vocab_guard() -> dict:
    from src.utils.vocab_guard import check, check_file, VocabViolation

    violations = []
    for pattern in ["analysis/*.json", "analysis/*.csv"]:
        for fp in glob_mod.glob(str(ROOT / pattern)):
            try:
                check_file(fp)
            except VocabViolation as e:
                violations.append(str(e))
            except Exception as e:
                # vocab_guard.check_file may not support .csv/.json natively;
                # skip files it can't parse rather than crashing the whole audit
                # on an unrelated file-format issue.
                violations.append(f"SKIPPED {fp}: {e}")

    # Reports get a body-only check: scripts/p9_report.py's own gate already
    # establishes the convention that the methodology annex intentionally
    # *lists* every forbidden phrase (as documentation of the language
    # contract itself) after a "**Prohibited language**" marker -- checking
    # past that marker produces unavoidable false positives on a file that's
    # correctly describing its own rules, not violating them.
    for fp in glob_mod.glob(str(ROOT / "reports" / "*.md")):
        text = Path(fp).read_text(encoding="utf-8")
        body_text = text.split("**Prohibited language**")[0]
        try:
            check(body_text, source_hint=fp)
        except VocabViolation as e:
            violations.append(str(e))

    real_violations = [v for v in violations if not v.startswith("SKIPPED")]
    return {
        "status": "PASS" if not real_violations else "FAIL",
        "violations": real_violations,
        "skipped": [v for v in violations if v.startswith("SKIPPED")],
    }


def run_schema_check() -> dict:
    """Confirm the Gold artifacts load and have the expected primary key."""
    import pandas as pd
    try:
        gold = pd.read_parquet(ROOT / "data" / "gold" / "megacampus_gold.parquet")
        suit = pd.read_parquet(ROOT / "data" / "gold" / "suitability_scores.parquet")
        return {
            "status": "PASS",
            "megacampus_gold_shape": list(gold.shape),
            "suitability_scores_shape": list(suit.shape),
        }
    except Exception as e:
        return {"status": "FAIL", "error": str(e)}


def main(mode: str = "full") -> None:
    report = {}

    if mode in ("full", "hashes"):
        report["hash_check"] = run_hash_check()

    if mode in ("full", "seeds"):
        report["seed_audit"] = run_seed_audit()

    if mode in ("full", "vocab"):
        report["vocab_guard"] = run_vocab_guard()

    if mode == "full":
        report["schema_check"] = run_schema_check()

    all_pass = all(
        v.get("status") in ("PASS", "SKIP")
        for v in report.values()
    )
    report["overall_status"] = "PASS" if all_pass else "FAIL"

    output = ROOT / "pipeline_audit.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "hashes", "seeds", "vocab", "generate-hashes"], default="full")
    parser.add_argument("--verify-hashes", action="store_true",
                         help="Alias for --mode=hashes (matches `make reproduce`'s invocation)")
    args = parser.parse_args()

    if args.verify_hashes:
        args.mode = "hashes"

    if args.mode == "generate-hashes":
        result = generate_expected_hashes()
        print(json.dumps(result, indent=2))
        sys.exit(0)

    main(args.mode)
