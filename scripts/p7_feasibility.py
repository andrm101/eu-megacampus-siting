"""
P7 — Input-Output & Feasibility Framework.

Three analytical components:

1. STRUCTURAL DEPENDENCY MATRIX (I-O analogue)
   An 8×8 matrix encoding the strength of structural complementarity between
   disruptor types (0=none, 1=weak synergy, 2=moderate co-dependency, 3=strong
   infrastructure dependency). Values are expert-coded from feature overlap
   analysis — not derived from economic I-O tables.

2. HARD FEASIBILITY GATES
   Per-type binary pass/fail gates based on absolute structural thresholds.
   Gates represent minimum viable preconditions: regions failing a hard gate
   are flagged as "precondition gap" irrespective of their suitability index.
   T6 (HALEU/Nuclear) carries a regulatory eligibility gate that is a hard stop.

3. DEVELOPMENT SCENARIO CLASSIFICATION
   Scenario A — Near-term structural candidates (2025–2030):
     Tier 1 (index >= 0.70) AND all hard gates passed.
   Scenario B — Medium-term pipeline (2030–2035):
     Tier 2 (0.60–0.70) AND primary gates passed AND structural readiness >= 0.70.
   Scenario C — Long-term investment pathway (2035–2040):
     Any shortlisted region where gap-closing investment is structurally feasible
     (SR >= 0.50 AND at most 2 hard gate failures with closeable gaps).

Outputs:
  analysis/p7_io_matrix.csv          — 8×8 dependency matrix with rationale
  analysis/p7_gate_results.csv       — per-region per-type gate pass/fail
  analysis/p7_scenario_assignments.csv
  analysis/p7_feasibility_report.json
  figures/p7_io_heatmap.png
  figures/p7_gate_waterfall.png      — funnel: shortlisted -> gates passed -> scenario A
  figures/p7_scenario_summary.png    — stacked bar: scenario A/B/C per type
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

np.random.seed(42)
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logging_config import pipeline_step  # noqa: E402

SCORES_PATH   = ROOT / "data" / "gold" / "suitability_scores.parquet"
GOLD_PATH     = ROOT / "data" / "gold" / "megacampus_gold.parquet"
SHORTLIST_CSV = ROOT / "analysis" / "p5_shortlist_tiered.csv"
ANALYSIS      = ROOT / "analysis"
FIGURES       = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

TIER1_THR = 0.70
TIER2_THR = 0.60
SR_NEAR   = 0.70   # structural readiness minimum for Scenario B
SR_LONG   = 0.50   # structural readiness minimum for Scenario C
MAX_CLOSEABLE_GATE_FAILS = 2  # Scenario C: max gate failures considered closeable

_BG     = "#0a0f1e"
_BLUE   = "#00a8ff"
_AMBER  = "#ff8c42"
_GREEN  = "#00cc7a"
_RED    = "#ff3355"
_PURPLE = "#a855f7"
_CYAN   = "#66d9e8"
_GOLD   = "#f5a623"
_PINK   = "#fbbf24"

TYPE_LABELS = {
    "T1": "AI / ML Hub",
    "T2": "Biotechnology / Life Sciences",
    "T3": "Semiconductors / Advanced Electronics",
    "T4": "Cleantech / Green Technology",
    "T5": "Hyperscale Data Centre Hub",
    "T6": "HALEU / Advanced Nuclear",
    "T7": "Deep-Tech Robotics Campus",
    "T8": "Quantum / Photonics Anchor",
}

TYPE_COLORS = {
    "T1": _BLUE, "T2": _GREEN, "T3": _AMBER, "T4": _CYAN,
    "T5": _GOLD, "T6": _RED,   "T7": _PURPLE, "T8": _PINK,
}

# ─── Structural dependency matrix (I-O analogue) ─────────────────────────────
# Value semantics:
#   0 = no structural dependency
#   1 = weak synergy (co-location beneficial; shared talent pool or markets)
#   2 = moderate co-dependency (shared infrastructure; overlapping feature requirements)
#   3 = strong infrastructure dependency (type in row requires type in column)
#
# Rationale for non-zero cells:
#   T1→T5: AI/ML compute clusters structurally depend on hyperscale data centre
#           infrastructure (low-cost power, high-bandwidth fibre, cooling).
#   T1→T2: AI/ML capability directly accelerates computational biology and
#           drug discovery pipelines; talent pool overlaps on computational skills.
#   T1→T8: Quantum ML is an emerging sub-discipline; quantum computing resources
#           (T8 infrastructure) are structurally complementary to frontier AI labs.
#   T2→T1: Biotech generates large omics datasets requiring AI/ML processing capacity.
#   T3→T7: Semiconductor fabs supply precision actuators, sensors, and embedded
#           chips to robotics manufacturing; process know-how overlaps.
#   T4→T5: Renewable energy generation (T4 strength) is the primary power supply
#           for hyperscale data centres seeking 24/7 carbon-free electricity.
#   T4→T6: Cleantech (renewables) and advanced nuclear provide complementary
#           dispatchable + variable generation for deep decarbonisation.
#   T5→T4: Data centre operators are major offtakers for PPA renewable contracts;
#           creates demand-side incentive for cleantech buildout.
#   T5→T6: SMR/advanced nuclear provides stable baseload for always-on data centres
#           where renewable intermittency is a grid-reliability constraint.
#   T6→T4: HALEU/SMR grid integration requires cleantech grid management and
#           storage technologies for load balancing.
#   T6→T8: Nuclear quantum sensing applications (isotope identification,
#           non-destructive evaluation) create direct T6→T8 demand linkage.
#   T7→T3: Robotics campus demand drives requirements for advanced electronics
#           (T3 C26 products) as embedded systems and precision components.
#   T8→T1: Quantum-accelerated optimisation and error-correction algorithms
#           are a structural input to next-generation AI/ML workloads.
#   T8→T6: Quantum sensing for nuclear non-proliferation and reactor diagnostics
#           is a structural application pathway for T8 research anchors.

IO_MATRIX = {
    #       T1  T2  T3  T4  T5  T6  T7  T8
    "T1": [ 0,  2,  0,  0,  3,  0,  0,  2 ],
    "T2": [ 2,  0,  0,  0,  0,  0,  0,  0 ],
    "T3": [ 0,  0,  0,  0,  0,  0,  3,  0 ],
    "T4": [ 0,  0,  0,  0,  2,  2,  0,  0 ],
    "T5": [ 0,  0,  0,  3,  0,  2,  0,  0 ],
    "T6": [ 0,  0,  0,  2,  0,  0,  0,  2 ],
    "T7": [ 0,  0,  3,  0,  0,  0,  0,  0 ],
    "T8": [ 2,  0,  0,  0,  0,  2,  0,  0 ],
}

IO_RATIONALE = {
    ("T1","T5"): "AI compute requires stable low-cost power + hyperscale connectivity",
    ("T1","T2"): "Shared computational talent; AI accelerates omics/drug-discovery pipelines",
    ("T1","T8"): "Quantum ML acceleration; frontier AI labs co-locate with quantum compute",
    ("T2","T1"): "Large omics datasets require AI/ML processing infrastructure",
    ("T3","T7"): "Semiconductor supply chain directly feeds robotics precision components",
    ("T4","T5"): "Renewable PPA offtake drives data-centre clean-power procurement",
    ("T4","T6"): "Complementary baseload (nuclear) + variable (renewables) grid mix",
    ("T5","T4"): "Data-centre renewable demand creates cleantech buildout incentive",
    ("T5","T6"): "SMR baseload removes renewable intermittency constraint for 24/7 DC ops",
    ("T6","T4"): "HALEU-SMR grid integration requires cleantech storage and management",
    ("T6","T8"): "Nuclear quantum sensing (isotope ID, reactor diagnostics) links T6-T8",
    ("T7","T3"): "Robotics embedded systems and precision actuators sourced from T3 fabs",
    ("T8","T1"): "Quantum optimisation algorithms structurally input to next-gen AI/ML",
    ("T8","T6"): "Quantum sensing applications in nuclear non-proliferation and diagnostics",
}


# ─── Hard feasibility gates ───────────────────────────────────────────────────
# Gate specification: (feature, operator, threshold_type, threshold_value_or_pct)
# threshold_type: "abs" = absolute value; "pct_ge" = >= Nth percentile of EU-242;
#                 "pct_le" = <= Nth percentile of EU-242; "bool" = must be 1
# hard_stop: True = regulatory/physical impossibility; False = structural gap

GATE_SPECS: dict[str, list[dict]] = {
    "T1": [
        {"label": "broadband_readiness",
         "feature": "ultrafast_broadband_pct", "operator": ">=",
         "threshold_type": "pct_ge", "threshold_pct": 25,
         "hard_stop": False,
         "rationale": "Minimum ultrafast broadband coverage for AI/ML infrastructure"},
        {"label": "talent_base",
         "feature": "hrst_per_1000", "operator": ">=",
         "threshold_type": "pct_ge", "threshold_pct": 25,
         "hard_stop": False,
         "rationale": "Human resources in S&T >= EU 25th percentile"},
    ],
    "T2": [
        {"label": "university_rd_base",
         "feature": "gerd_hes_pct_gdp", "operator": ">=",
         "threshold_type": "pct_ge", "threshold_pct": 25,
         "hard_stop": False,
         "rationale": "HES R&D spending >= EU 25th percentile — minimum academic pipeline"},
        {"label": "talent_base",
         "feature": "hrst_per_1000", "operator": ">=",
         "threshold_type": "pct_ge", "threshold_pct": 25,
         "hard_stop": False,
         "rationale": "Life-sciences talent pool minimum"},
    ],
    "T3": [
        {"label": "energy_cost_competitive",
         "feature": "electricity_price_eur_kwh", "operator": "<=",
         "threshold_type": "pct_le", "threshold_pct": 75,
         "hard_stop": False,
         "rationale": "Industrial electricity price <= EU 75th percentile (fab opex constraint)"},
        {"label": "business_environment",
         "feature": "doing_business_score", "operator": ">=",
         "threshold_type": "pct_ge", "threshold_pct": 25,
         "hard_stop": False,
         "rationale": "Regulatory environment >= EU 25th percentile"},
    ],
    "T4": [
        {"label": "renewable_share_minimum",
         "feature": "renewable_energy_share_pct", "operator": ">=",
         "threshold_type": "abs", "threshold_value": 20.0,
         "hard_stop": False,
         "rationale": "National renewable share >= 20% — minimum grid decarbonisation signal"},
    ],
    "T5": [
        {"label": "energy_cost_cap",
         "feature": "electricity_price_eur_kwh", "operator": "<=",
         "threshold_type": "pct_le", "threshold_pct": 60,
         "hard_stop": False,
         "rationale": "Hyperscale DC electricity cost <= EU 60th percentile (PUE economics)"},
        {"label": "water_stress_cap",
         "feature": "water_exploitation_index", "operator": "<=",
         "threshold_type": "pct_le", "threshold_pct": 75,
         "hard_stop": False,
         "rationale": "Water exploitation index <= EU 75th percentile (cooling water access)"},
        {"label": "grid_capacity_minimum",
         "feature": "ntc_import_mw", "operator": ">=",
         "threshold_type": "pct_ge", "threshold_pct": 25,
         "hard_stop": False,
         "rationale": "Country NTC import capacity >= EU 25th percentile"},
    ],
    "T6": [
        {"label": "regulatory_eligibility",
         "feature": "haleu_smr_eligible_int", "operator": "==",
         "threshold_type": "bool", "threshold_value": 1.0,
         "hard_stop": True,
         "rationale": "HARD STOP: national nuclear policy must permit HALEU/SMR deployment"},
        {"label": "grid_capacity_minimum",
         "feature": "ntc_import_mw", "operator": ">=",
         "threshold_type": "pct_ge", "threshold_pct": 25,
         "hard_stop": False,
         "rationale": "Grid interconnection capacity for nuclear output export"},
    ],
    "T7": [
        {"label": "talent_base",
         "feature": "score_talent", "operator": ">=",
         "threshold_type": "pct_ge", "threshold_pct": 25,
         "hard_stop": False,
         "rationale": "Robotics talent pool >= EU 25th percentile"},
        {"label": "business_environment",
         "feature": "doing_business_score", "operator": ">=",
         "threshold_type": "pct_ge", "threshold_pct": 25,
         "hard_stop": False,
         "rationale": "Regulatory environment for advanced manufacturing"},
    ],
    "T8": [
        {"label": "national_quantum_programme",
         "feature": "quantum_programme_tier", "operator": ">",
         "threshold_type": "abs", "threshold_value": 0.0,
         "hard_stop": False,
         "rationale": "Country must have a national quantum programme (tier >= 1)"},
    ],
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _get_col(gold: pd.DataFrame, feat: str) -> pd.Series:
    if feat not in gold.columns:
        return pd.Series(np.nan, index=gold.index)
    col = gold[feat]
    if col.dtype == bool or col.dtype == object:
        col = col.astype(float)
    return pd.to_numeric(col, errors="coerce").fillna(
        pd.to_numeric(gold[feat], errors="coerce").median()
        if gold[feat].dtype != object else 0.0
    )


def _prepare_gold(gold: pd.DataFrame) -> pd.DataFrame:
    g = gold.copy()
    if "nuclear_capacity_mw" in g.columns:
        g["nuclear_capacity_mw_log1p"] = np.log1p(g["nuclear_capacity_mw"].fillna(0))
    if "haleu_smr_eligible" in g.columns:
        g["haleu_smr_eligible_int"] = g["haleu_smr_eligible"].astype(float)
    if "quantum_programme_tier" in g.columns:
        g["quantum_programme_tier"] = pd.to_numeric(
            g["quantum_programme_tier"], errors="coerce").fillna(0) / 3.0
    if "is_transition_region" in g.columns:
        g["is_transition_region"] = g["is_transition_region"].astype(float)
    return g


def compute_thresholds(gold: pd.DataFrame) -> dict[str, float]:
    """Pre-compute all percentile thresholds from the EU-242 Gold sample."""
    thresholds: dict[str, float] = {}
    all_feats = set()
    for specs in GATE_SPECS.values():
        for g in specs:
            all_feats.add(g["feature"])

    for feat in all_feats:
        col = _get_col(gold, feat)
        for pct in [25, 40, 60, 75]:
            thresholds[f"{feat}_pct{pct}"] = float(np.nanpercentile(col.dropna(), pct))
    return thresholds


def evaluate_gates(gold: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Return a DataFrame with gate pass/fail per region per type."""
    rows: dict[str, dict] = {nuts2: {} for nuts2 in gold.index}

    for tid, specs in GATE_SPECS.items():
        for g in specs:
            feat   = g["feature"]
            op     = g["operator"]
            tt     = g["threshold_type"]
            label  = g["label"]
            col    = _get_col(gold, feat)
            hs     = g["hard_stop"]

            if tt == "abs":
                thr = float(g["threshold_value"])
            elif tt == "bool":
                thr = float(g["threshold_value"])
            elif tt.startswith("pct_"):
                pct = g["threshold_pct"]
                thr = thresholds[f"{feat}_pct{pct}"]
            else:
                thr = float(g.get("threshold_value", 0))

            if op == ">=":
                passed = col >= thr
            elif op == "<=":
                passed = col <= thr
            elif op == "==":
                passed = col == thr
            elif op == ">":
                passed = col > thr
            else:
                passed = col >= thr

            col_name = f"gate_{tid}_{label}"
            for nuts2 in gold.index:
                rows[nuts2][col_name]          = bool(passed.loc[nuts2])
                rows[nuts2][f"{col_name}_hs"]  = hs
                rows[nuts2][f"{col_name}_thr"] = round(thr, 4)
                rows[nuts2][f"{col_name}_val"] = round(float(col.loc[nuts2]), 4)

    return pd.DataFrame.from_dict(rows, orient="index")


def assign_scenarios(
        scores: pd.DataFrame,
        gate_df: pd.DataFrame,
        shortlist_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Classify each region × type into Scenario A, B, C, or None.

    Scenario A: Tier 1 AND all hard gates passed (includes all gates).
    Scenario B: Tier 2 AND primary (hard_stop=False) gates passed AND SR >= SR_NEAR.
    Scenario C: Any shortlisted AND at most MAX_CLOSEABLE_GATE_FAILS soft gate failures
                AND SR >= SR_LONG.
    """
    scenario_rows: dict[str, dict] = {n: {} for n in scores.index}

    for tid in TYPE_LABELS:
        score_col = f"suitability_{tid}"
        tier_col  = f"tier_{tid}"
        sr_col    = f"sr_{tid}"

        tier = shortlist_df[tier_col] if tier_col in shortlist_df.columns \
            else pd.Series("", index=scores.index)
        sr   = shortlist_df[sr_col] if sr_col in shortlist_df.columns \
            else pd.Series(1.0, index=scores.index)
        sc   = scores[score_col] if score_col in scores.columns \
            else pd.Series(0.0, index=scores.index)

        # Gather gate columns for this type
        gate_cols     = [c for c in gate_df.columns if c.startswith(f"gate_{tid}_") and not c.endswith(("_hs","_thr","_val"))]
        hard_stop_cols = [c for c in gate_cols
                          if gate_df.get(f"{c}_hs", pd.Series([False]*len(gate_df))).iloc[0]]

        for nuts2 in scores.index:
            s = scenario_rows[nuts2]
            t   = tier.loc[nuts2] if nuts2 in tier.index else ""
            r   = float(sr.loc[nuts2]) if nuts2 in sr.index else 1.0
            idx = float(sc.loc[nuts2]) if nuts2 in sc.index else 0.0

            all_gates   = all(gate_df.loc[nuts2, c] for c in gate_cols if nuts2 in gate_df.index) \
                          if gate_cols else True
            hs_passed   = all(gate_df.loc[nuts2, c] for c in hard_stop_cols if nuts2 in gate_df.index) \
                          if hard_stop_cols else True
            soft_fails  = sum(1 for c in gate_cols if nuts2 in gate_df.index
                              and not gate_df.loc[nuts2, c]
                              and not gate_df.get(f"{c}_hs", pd.Series([False]*len(gate_df))).get(nuts2, False)) \
                          if gate_cols else 0

            if t == "T1_high_index" and all_gates:
                scenario = "A_near_term"
            elif t == "T2_candidate" and hs_passed and r >= SR_NEAR:
                scenario = "B_medium_term"
            elif t in ("T1_high_index", "T2_candidate") and hs_passed \
                    and soft_fails <= MAX_CLOSEABLE_GATE_FAILS and r >= SR_LONG:
                scenario = "C_long_term"
            else:
                scenario = "none"

            s[f"scenario_{tid}"] = scenario
            s[f"n_gates_{tid}"]  = len(gate_cols)
            s[f"gates_passed_{tid}"] = sum(1 for c in gate_cols if nuts2 in gate_df.index
                                           and gate_df.loc[nuts2, c]) if gate_cols else len(gate_cols)

    return pd.DataFrame.from_dict(scenario_rows, orient="index")


# ─── I-O matrix as DataFrame ─────────────────────────────────────────────────

def build_io_dataframe() -> pd.DataFrame:
    tids = list(TYPE_LABELS.keys())
    matrix = pd.DataFrame(IO_MATRIX, index=tids, columns=tids)
    return matrix


# ─── Figures ─────────────────────────────────────────────────────────────────

def fig_io_heatmap(io_df: pd.DataFrame) -> None:
    labels = [f"{t}\n{TYPE_LABELS[t][:20]}" for t in io_df.index]
    fig, ax = plt.subplots(figsize=(10, 9), facecolor=_BG)
    ax.set_facecolor("#0d1529")

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "io", ["#0d1529", "#1e3a5f", _BLUE, _AMBER, _RED])
    im = ax.imshow(io_df.values, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0,1,2,3])
    cbar.ax.set_yticklabels(["0 None","1 Synergy","2 Co-dep","3 Strong dep"],
                             color="#c0d0e8", fontsize=7.5)
    cbar.ax.tick_params(colors="#c0d0e8")

    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels(labels, rotation=45, ha="right", color="#c0d0e8", fontsize=8)
    ax.set_yticklabels(labels, color="#c0d0e8", fontsize=8)
    ax.set_xlabel("Column = structural dependency target", color="#c0d0e8", fontsize=8)
    ax.set_ylabel("Row = structural dependency source", color="#c0d0e8", fontsize=8)

    for i in range(8):
        for j in range(8):
            v = io_df.values[i, j]
            if v > 0:
                key = (list(TYPE_LABELS.keys())[i], list(TYPE_LABELS.keys())[j])
                short = IO_RATIONALE.get(key, "")[:30]
                ax.text(j, i, f"{int(v)}", ha="center", va="center",
                        color="white", fontsize=10, fontweight="bold")

    ax.set_title("DISRUPTOR ECOSYSTEM STRUCTURAL DEPENDENCY MATRIX\n"
                 "Row → Column: strength of structural co-dependency",
                 color=_BLUE, fontsize=11, fontfamily="monospace", pad=14)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e3050")
    ax.tick_params(colors="#c0d0e8")
    plt.tight_layout()
    out = FIGURES / "p7_io_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print(f"    Saved: {out.relative_to(ROOT)}")


def fig_gate_waterfall(scores: pd.DataFrame, gate_df: pd.DataFrame,
                       scenario_df: pd.DataFrame) -> None:
    """Per-type funnel: shortlisted → hard gates passed → Scenario A."""
    tids   = list(TYPE_LABELS.keys())
    n_sl   = []  # shortlisted (Tier1+Tier2)
    n_hs   = []  # hard gates passed
    n_all  = []  # all gates passed
    n_scA  = []  # scenario A

    for tid in tids:
        score_col = f"suitability_{tid}"
        sc = scores[score_col] if score_col in scores.columns else pd.Series(0.0, index=scores.index)
        sl = (sc >= TIER2_THR).sum()

        gate_cols   = [c for c in gate_df.columns if c.startswith(f"gate_{tid}_")
                       and not c.endswith(("_hs","_thr","_val"))]
        hs_cols     = [c for c in gate_cols
                       if any(gate_df.get(f"{c}_hs",
                              pd.Series([False]*len(gate_df))).values)]

        shortlisted_idx = sc.index[sc >= TIER2_THR]
        hs_pass = len(shortlisted_idx)
        all_pass = len(shortlisted_idx)
        if hs_cols and len(shortlisted_idx) > 0:
            hs_mask = gate_df.loc[shortlisted_idx, hs_cols].all(axis=1)
            hs_pass = int(hs_mask.sum())
        if gate_cols and len(shortlisted_idx) > 0:
            all_mask = gate_df.loc[shortlisted_idx, gate_cols].all(axis=1)
            all_pass = int(all_mask.sum())

        scA_col = f"scenario_{tid}"
        scA = int((scenario_df[scA_col] == "A_near_term").sum()) if scA_col in scenario_df.columns else 0

        n_sl.append(sl); n_hs.append(hs_pass); n_all.append(all_pass); n_scA.append(scA)

    x   = np.arange(len(tids))
    w   = 0.18
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=_BG)
    ax.set_facecolor("#0d1529")

    ax.bar(x - 1.5*w, n_sl,  width=w, label="Shortlisted (≥0.60)", color=_BLUE,  alpha=0.8)
    ax.bar(x - 0.5*w, n_hs,  width=w, label="Hard gates passed",    color=_GREEN, alpha=0.8)
    ax.bar(x + 0.5*w, n_all, width=w, label="All gates passed",      color=_AMBER, alpha=0.8)
    ax.bar(x + 1.5*w, n_scA, width=w, label="Scenario A (near-term)",color=_RED,  alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(tids, color="#c0d0e8", fontsize=10, fontfamily="monospace")
    ax.set_ylabel("Number of NUTS2 regions", color="#c0d0e8", fontsize=9)
    ax.set_title("FEASIBILITY FUNNEL — SHORTLIST → GATE CHECKS → SCENARIO A\n"
                 "EU NUTS2 regions by type",
                 color=_BLUE, fontsize=11, fontfamily="monospace", pad=10)
    ax.tick_params(colors="#c0d0e8", labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e3050")
    leg = ax.legend(fontsize=8.5, framealpha=0.3, facecolor="#0d1529",
                    edgecolor="#1e3050", labelcolor="#c0d0e8")
    plt.tight_layout()
    out = FIGURES / "p7_gate_waterfall.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print(f"    Saved: {out.relative_to(ROOT)}")


def fig_scenario_summary(scenario_df: pd.DataFrame) -> None:
    """Stacked bar: scenario A/B/C counts per type."""
    tids = list(TYPE_LABELS.keys())
    scA = []; scB = []; scC = []
    for tid in tids:
        col = f"scenario_{tid}"
        if col not in scenario_df.columns:
            scA.append(0); scB.append(0); scC.append(0)
            continue
        scA.append((scenario_df[col] == "A_near_term").sum())
        scB.append((scenario_df[col] == "B_medium_term").sum())
        scC.append((scenario_df[col] == "C_long_term").sum())

    x = np.arange(len(tids))
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=_BG)
    ax.set_facecolor("#0d1529")

    b1 = ax.bar(x, scA, label="A — Near-term (2025–30)",   color=_GREEN,  alpha=0.9)
    b2 = ax.bar(x, scB, bottom=scA,
                label="B — Medium-term (2030–35)",          color=_AMBER,  alpha=0.8)
    b3 = ax.bar(x, scC, bottom=[a+b for a,b in zip(scA,scB)],
                label="C — Long-term (2035–40)",            color=_PURPLE, alpha=0.7)

    for i, (a, b, c) in enumerate(zip(scA, scB, scC)):
        if a > 0: ax.text(x[i], a/2, str(a), ha="center", va="center",
                          color=_BG, fontsize=9, fontfamily="monospace", fontweight="bold")
        if b > 0: ax.text(x[i], a+b/2, str(b), ha="center", va="center",
                          color="#c0d0e8", fontsize=8.5, fontfamily="monospace")
        if c > 0: ax.text(x[i], a+b+c/2, str(c), ha="center", va="center",
                          color="#c0d0e8", fontsize=8.5, fontfamily="monospace")

    ax.set_xticks(x)
    ax.set_xticklabels(tids, color="#c0d0e8", fontsize=10.5, fontfamily="monospace")
    ax.set_ylabel("Number of EU NUTS2 regions", color="#c0d0e8", fontsize=9)
    ax.set_title("DEVELOPMENT SCENARIO CLASSIFICATION — EU NUTS2\n"
                 "Regions by type and structural readiness horizon",
                 color=_BLUE, fontsize=11, fontfamily="monospace", pad=10)
    ax.tick_params(colors="#c0d0e8", labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e3050")
    ax.legend(fontsize=8.5, framealpha=0.3, facecolor="#0d1529",
              edgecolor="#1e3050", labelcolor="#c0d0e8")
    plt.tight_layout()
    out = FIGURES / "p7_scenario_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close()
    print(f"    Saved: {out.relative_to(ROOT)}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    with pipeline_step("p7_load"):
        scores = pd.read_parquet(SCORES_PATH)
        gold   = _prepare_gold(pd.read_parquet(GOLD_PATH))
        short  = pd.read_csv(SHORTLIST_CSV, index_col=0)

    with pipeline_step("p7_io_matrix"):
        io_df = build_io_dataframe()
        io_rows = []
        for src in TYPE_LABELS:
            for tgt in TYPE_LABELS:
                v = IO_MATRIX[src][list(TYPE_LABELS.keys()).index(tgt)]
                rat = IO_RATIONALE.get((src, tgt), "")
                io_rows.append({"from_type": src, "to_type": tgt,
                                 "dependency_strength": v, "rationale": rat})
        pd.DataFrame(io_rows).to_csv(ANALYSIS / "p7_io_matrix.csv", index=False)

    with pipeline_step("p7_gates"):
        thresholds = compute_thresholds(gold)
        gate_df    = evaluate_gates(gold, thresholds)
        gate_df.to_csv(ANALYSIS / "p7_gate_results.csv")

    with pipeline_step("p7_scenarios"):
        scenario_df = assign_scenarios(scores, gate_df, short)
        scenario_df.to_csv(ANALYSIS / "p7_scenario_assignments.csv")

    with pipeline_step("p7_report"):
        report: dict = {
            "io_matrix_max": int(io_df.values.max()),
            "io_total_links": int((io_df.values > 0).sum()),
            "io_strong_links": int((io_df.values == 3).sum()),
            "scenario_thresholds": {
                "A_tier1_min": TIER1_THR,
                "B_tier2_min": TIER2_THR, "B_sr_min": SR_NEAR,
                "C_sr_min": SR_LONG, "C_max_gate_fails": MAX_CLOSEABLE_GATE_FAILS,
            },
            "types": {},
        }
        for tid in TYPE_LABELS:
            scol  = f"scenario_{tid}"
            gcols = [c for c in gate_df.columns if c.startswith(f"gate_{tid}_")
                     and not c.endswith(("_hs","_thr","_val"))]
            scA = int((scenario_df[scol] == "A_near_term").sum()) if scol in scenario_df.columns else 0
            scB = int((scenario_df[scol] == "B_medium_term").sum()) if scol in scenario_df.columns else 0
            scC = int((scenario_df[scol] == "C_long_term").sum()) if scol in scenario_df.columns else 0
            all_pass = int(gate_df[gcols].all(axis=1).sum()) if gcols else 242
            report["types"][tid] = {
                "label": TYPE_LABELS[tid],
                "n_gates": len(gcols),
                "n_hard_stops": sum(1 for g in GATE_SPECS.get(tid, []) if g["hard_stop"]),
                "regions_all_gates_passed": all_pass,
                "scenario_A": scA, "scenario_B": scB, "scenario_C": scC,
            }
        with open(ANALYSIS / "p7_feasibility_report.json", "w") as f:
            json.dump(report, f, indent=2)

    with pipeline_step("p7_figures"):
        fig_io_heatmap(io_df)
        fig_gate_waterfall(scores, gate_df, scenario_df)
        fig_scenario_summary(scenario_df)

    # ── Gate check ────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("P7 FEASIBILITY FRAMEWORK — GATE CHECK")
    print("=" * 70)
    print(f"  I-O matrix: {report['io_total_links']} dependency links, "
          f"{report['io_strong_links']} strong (value=3)")
    print()
    print(f"  {'Type':<6} {'Gates':>6} {'HardStop':>9} {'AllPass':>8} "
          f"{'ScenA':>7} {'ScenB':>7} {'ScenC':>7}")
    print(f"  {'-'*6} {'-'*6} {'-'*9} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")
    for tid, r in report["types"].items():
        print(f"  {tid:<6} {r['n_gates']:>6} {r['n_hard_stops']:>9} "
              f"{r['regions_all_gates_passed']:>8} "
              f"{r['scenario_A']:>7} {r['scenario_B']:>7} {r['scenario_C']:>7}")
    print()

    all_types_have_scA = all(
        report["types"][t]["scenario_A"] >= 1 for t in TYPE_LABELS
    )
    gate_pass = all_types_have_scA

    print(f"GATE_P7={'PASS' if gate_pass else 'FAIL'}")
    if not gate_pass:
        missing = [t for t in TYPE_LABELS if report["types"][t]["scenario_A"] == 0]
        print(f"  REASON: no Scenario A regions for {missing}")
    print("=" * 70)

    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
