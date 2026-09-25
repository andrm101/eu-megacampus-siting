"""
Vocabulary guard for EU-MegaCampus-Siting.

This project uses a strategic intelligence framing — composite indices,
ranked shortlists, and structural assessments are permitted. The guard
blocks only analytically dishonest or legally problematic patterns:
absolute unqualified superlatives, causal claims unsupported by design,
and deterministic investment guarantees.
"""

import re
import sys
from pathlib import Path

# Patterns that are ALWAYS forbidden regardless of project framing
FORBIDDEN_PATTERNS: list[tuple[str, str]] = [
    # Absolute unqualified superlatives with no index/metric qualifier
    (r"\bthe\s+best\s+region\b", "unqualified superlative: 'the best region'"),
    (r"\boptimal\s+location\b", "unqualified superlative: 'optimal location'"),
    # Hard causal claims (only allow if RCT/DiD/IV established in context)
    (r"\bcauses\s+(?:higher|lower|more|less|greater|increased|decreased)\b", "causal: 'causes [outcome]'"),
    (r"\beffect\s+of\s+\w+\s+on\b", "causal: 'effect of X on Y'"),
    # Deterministic investment guarantees
    (r"\bguarantee[sd]?\s+(?:returns?|success|growth|performance)\b", "guarantee: investment certainty claim"),
    (r"\bwill\s+(?:succeed|outperform|deliver|generate)\b", "deterministic investment outcome"),
    # Legally problematic direct targeting
    (r"\b(?:Palantir|Anduril|Microsoft|Amazon|Google|Meta|Apple)\s+should\s+locate\b", "direct company siting directive"),
    # Composite ranking (use 'composite index' or 'suitability index' instead)
    (r"\bcomposite\s+ranking\b", "prohibited term: use 'composite index' or 'suitability index'"),
    (r"\bsite\s+selection\s+score\b", "prohibited term: use 'suitability index'"),
]

# Patterns that are PERMITTED in this project (for documentation purposes)
# - "top N regions by SC1 index" — OK (quantified, index-relative)
# - "structurally favourable" — OK (relative, hedged)
# - "above threshold" / "shortlisted" — OK (rule-based, transparent)
# - "ranked by composite index" — OK (explicit, not 'composite ranking')
# - "high-index regions" — OK (index-relative)


class VocabViolation(ValueError):
    pass


def check(text: str, source_hint: str = "<string>") -> None:
    violations: list[str] = []
    for pattern, label in FORBIDDEN_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            violations.append(f"  [{label}] matched: {matches[:3]}")
    if violations:
        msg = f"VocabGuard violations in {source_hint}:\n" + "\n".join(violations)
        raise VocabViolation(msg)


def check_file(path: str | Path) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"vocab_guard: file not found: {p}")
    text = p.read_text(encoding="utf-8", errors="replace")
    check(text, source_hint=str(p))


def check_dataframe_strings(df, source_hint: str = "<dataframe>") -> None:
    import pandas as pd
    for col in df.select_dtypes(include="object").columns:
        for val in df[col].dropna().astype(str):
            check(val, source_hint=f"{source_hint}[{col}]")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vocab_guard.py <file_path>")
        sys.exit(1)
    try:
        check_file(sys.argv[1])
        print(f"vocab_guard: PASS — {sys.argv[1]}")
    except VocabViolation as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
