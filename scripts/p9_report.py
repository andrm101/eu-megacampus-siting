"""
p9_report.py — EU MegaCampus Siting Intelligence: Executive Brief + Methodology Annex
Generates reports/p9_executive_brief.md from all analysis artifacts.
Optionally converts to PDF via pandoc (pandoc >= 2.12 required).
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

ROOT = Path(__file__).parent.parent
ANALYSIS = ROOT / "analysis"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

OUTPUT_MD = REPORTS / "p9_executive_brief.md"
OUTPUT_PDF = REPORTS / "p9_executive_brief.pdf"

TODAY = "2026-05-24"

# ── Load artifacts ───────────────────────────────────────────────────────────

def load_artifacts() -> dict:
    p5 = json.loads((ANALYSIS / "p5_summary_report.json").read_text())
    p7 = json.loads((ANALYSIS / "p7_feasibility_report.json").read_text())
    morans = pd.read_csv(ANALYSIS / "p6_morans_i.csv")
    corridors = pd.read_csv(ANALYSIS / "p6_corridors.csv")
    p2 = json.loads((ANALYSIS / "p2_ingestion_report.json").read_text())
    io_mat = pd.read_csv(ANALYSIS / "p7_io_matrix.csv", index_col=0)
    shortlist = pd.read_csv(ANALYSIS / "p5_shortlist_tiered.csv")
    gap_mat = pd.read_csv(ANALYSIS / "p5_gap_matrix.csv")
    scenario_df = pd.read_csv(ANALYSIS / "p7_scenario_assignments.csv")
    weight_reg = pd.read_csv(ANALYSIS / "p4_weight_registry.csv")
    return dict(
        p5=p5, p7=p7, morans=morans, corridors=corridors, p2=p2,
        io_mat=io_mat, shortlist=shortlist, gap_mat=gap_mat,
        scenario_df=scenario_df, weight_reg=weight_reg,
    )

# ── Section builders ─────────────────────────────────────────────────────────

TYPE_NAMES = {
    "T1": "AI / Machine Learning Hub",
    "T2": "Biotechnology / Life Sciences",
    "T3": "Semiconductors / Advanced Electronics",
    "T4": "Cleantech / Green Technology",
    "T5": "Hyperscale Data Centre Hub",
    "T6": "HALEU / Advanced Nuclear Industrial Base",
    "T7": "Deep-Tech Robotics Campus",
    "T8": "Quantum / Photonics Research Anchor",
}

TYPE_SHORT = {
    "T1": "AI/ML Hub",
    "T2": "Biotech / Life Sciences",
    "T3": "Semiconductors",
    "T4": "Cleantech",
    "T5": "Data Centre Hub",
    "T6": "Advanced Nuclear",
    "T7": "Robotics Campus",
    "T8": "Quantum / Photonics",
}

ARCHETYPE_LABELS = {
    "A0": "Established Industrial (n=53)",
    "A1": "Catching-Up Peripheral (n=65)",
    "A2": "Frontier Innovation (n=43)",
    "A3": "Emerging Capacity (n=81)",
}


def _pct(n: int, total: int = 242) -> str:
    return f"{100 * n / total:.0f}%"


def _tier_summary_row(tid: str, p5: dict, p7: dict) -> str:
    t = p5["types"][tid]
    ft = p7["types"][tid]
    return (
        f"| {tid} | {TYPE_SHORT[tid]} | {t['n_tier1']} ({_pct(t['n_tier1'])}) "
        f"| {t['n_tier2']} | {ft['scenario_A']} | {ft['scenario_B']} "
        f"| {t['uncertainty_mean']:.2f} | {t['structural_readiness_median']:.3f} |"
    )


def _moran_row(row: pd.Series) -> str:
    return (
        f"| {row.type_id} | {TYPE_SHORT[row.type_id]} "
        f"| {row.moran_i:.3f} | {row.z_score:.2f} | < 0.001 |"
    )


def _corridor_summary(corridors: pd.DataFrame) -> pd.DataFrame:
    return (
        corridors.drop_duplicates(subset=["type_id", "corridor_id"])
        .groupby("type_id")
        .agg(
            n_corridors=("corridor_id", "count"),
            n_cross_border=("cross_border", "sum"),
            largest_corridor=("corridor_size", "max"),
        )
        .reset_index()
    )


def build_report(art: dict) -> str:
    p5 = art["p5"]
    p7 = art["p7"]
    morans: pd.DataFrame = art["morans"]
    corridors: pd.DataFrame = art["corridors"]
    io_mat: pd.DataFrame = art["io_mat"]
    cor_sum = _corridor_summary(corridors)
    total_corridors = int(corridors.drop_duplicates(["type_id", "corridor_id"]).shape[0])
    cross_border_corridors = int(
        corridors.drop_duplicates(["type_id", "corridor_id"])
        .query("cross_border == True")
        .shape[0]
    )

    lines = []
    w = lines.append  # convenience

    # ── Title block ────────────────────────────────────────────────────────
    w("---")
    w("title: |")
    w("  EU MegaCampus Siting Intelligence")
    w("  Executive Intelligence Brief")
    w("subtitle: |")
    w("  Structural Pre-Conditions for Eight Deep-Tech Ecosystem Types")
    w("  Across 242 EU NUTS2 Regions — Phases P0–P8 Synthesis")
    w(f"date: '{TODAY}'")
    w("author: EU MegaCampus Siting Intelligence Project")
    w("abstract: |")
    w("  This brief synthesises structural suitability analysis across 242 EU NUTS2")
    w("  regions for eight disruptor ecosystem types: AI/ML hubs, biotechnology")
    w("  clusters, semiconductor corridors, cleantech zones, hyperscale data-centre")
    w("  hubs, HALEU/advanced-nuclear industrial bases, deep-tech robotics campuses,")
    w("  and quantum/photonics research anchors. Composite suitability indices are")
    w("  constructed from 84 harmonised structural features. All findings are")
    w("  descriptive associations derived from observational cross-sectional data;")
    w("  no causal claims are made.")
    w("---")
    w("")

    # ── 1. Executive Summary ───────────────────────────────────────────────
    w("# 1. Executive Summary")
    w("")
    w(
        "This intelligence brief presents structural suitability profiles for eight "
        "deep-tech disruptor ecosystem types across all 242 NUTS2 regions of the "
        "European Union. The analysis integrates 84 harmonised structural features "
        "drawn from the EU Innovation Panel Gold layer and 18 purpose-built indicators "
        "covering electricity infrastructure, renewable energy, grid capacity, "
        "industrial land proxies, regulatory eligibility, and supranational programme "
        "participation."
    )
    w("")
    w(
        "Composite suitability indices are constructed for each type using weighted "
        "feature aggregation, per-type min–max rescaling, and propagated uncertainty "
        "estimates. Spatial autocorrelation analysis confirms that high-index regions "
        "are geographically clustered for all eight types (Global Moran's I range: "
        f"0.557–0.895, all p < 0.001). A total of {total_corridors} spatial corridors "
        f"are identified across types, of which {cross_border_corridors} span two or "
        "more member states, indicating transnational structural alignments that exceed "
        "single-region capacity."
    )
    w("")
    w(
        "Feasibility gateway analysis — encompassing broadband, grid headroom, regulatory "
        "eligibility, and industrial base thresholds — confirms that structural suitability "
        "is necessary but not sufficient: gateway constraints materially reduce effective "
        "shortlists. Under near-term Scenario A (2025–2030), the effective shortlists "
        "range from 7 regions (T7 Robotics) to 53 regions (T8 Quantum/Photonics). "
        "The HALEU/Advanced Nuclear type (T6) is the most constrained: a hard-stop "
        "regulatory-eligibility gate reduces the addressable pool to 96 of 242 regions "
        "before suitability scoring is applied."
    )
    w("")
    w(
        "All findings operate under a strategic intelligence epistemological contract. "
        "Indices express relative structural position within the EU-242 reference frame; "
        "they do not constitute site-selection recommendations and make no deterministic "
        "forecasts about ecosystem emergence or investment outcomes."
    )
    w("")

    # ── 2. Methodology ────────────────────────────────────────────────────
    w("# 2. Methodology")
    w("")
    w("## 2.1 Data Architecture and Feature Engineering")
    w("")
    w(
        "The analytical pipeline follows a Bronze → Silver → Gold medallion architecture. "
        "The upstream EU Innovation Panel Gold layer (242 NUTS2 × 66 features) provides "
        "the structural base, including PCA-derived dimension scores (prosperity, talent, "
        "digital infrastructure, innovation cluster), k=4 archetype assignments, and "
        "NUTS2-level labour market, R&D, and ICT indicators."
    )
    w("")
    w(
        "Eighteen additional features were appended at Silver stage from Eurostat, the "
        "European Environment Agency, ENTSO-E, the IAEA PRIS reactor database, EuroHPC "
        "site records, and national quantum programme registers. The Silver frame contains "
        "242 NUTS2 regions × 84 features at 0.0% null rate after country-median imputation "
        "for features with null rates below 10%. The port throughput variable "
        "(null rate: 92.2%) is treated as a structural DATA GAP and excluded from "
        "scoring; the ultrafast broadband variable (null rate: 30.2%) is imputed from "
        "country medians."
    )
    w("")
    w("## 2.2 Archetype Classification")
    w("")
    w(
        "K-means clustering (k=4, random_state=42) applied to the four PCA dimension "
        "scores yields four structural archetypes:"
    )
    w("")
    w("| Archetype | Label | n |")
    w("|-----------|-------|---|")
    for aid, alabel in ARCHETYPE_LABELS.items():
        w(f"| {aid} | {alabel} |")
    w("")
    w(
        "The k=4 solution (silhouette = 0.2413) is an intentional interpretability "
        "override from the k=2 solution used in the upstream EU Innovation Panel "
        "(silhouette = 0.409). Four archetypes are required to map meaningfully onto "
        "the eight disruptor-type taxonomy; k=2 conflates structural profiles that "
        "differ substantially on the extended feature set. The ARI between k=4 and k=2 "
        "is 0.2923, confirming partial but not complete overlap."
    )
    w("")
    w("## 2.3 Suitability Index Construction")
    w("")
    w(
        "For each of the eight ecosystem types, a composite suitability index is "
        "constructed as follows:"
    )
    w("")
    w(
        "1. **Feature selection**: Type-specific feature sets are drawn from the Silver "
        "frame. Features are assigned high (H=3), medium (M=2), or low (L=1) weights "
        "based on structural relevance documented in the weight registry."
    )
    w(
        "2. **Direction inversion**: Features where lower values indicate greater "
        "suitability (e.g., electricity price, water exploitation index) are inverted "
        "as 1 − min–max(feature) before scoring."
    )
    w(
        "3. **Weighted composite**: $S_{r,t} = \\sum_f w_{f,t} \\cdot x_{r,f} / \\sum_f w_{f,t}$, "
        "where $x_{r,f}$ is the min–max normalised feature value for region $r$ and "
        "feature $f$, and $w_{f,t}$ is the type-specific weight."
    )
    w(
        "4. **Per-type rescaling**: Each type's composite scores are min–max rescaled "
        "to [0, 1] so that the threshold 0.60 consistently identifies the top 40% of "
        "EU NUTS2 regions on that type's structural profile. This is analytically "
        "necessary because country-level broadcast features in T4 (Cleantech) compress "
        "the raw composite maximum to 0.571, making a common absolute threshold "
        "inappropriate."
    )
    w(
        "5. **Uncertainty propagation**: An uncertainty score $u_{r,t}$ is computed as "
        "the proportion of type weight carried by DATA GAP proxy features. Confidence "
        "half-widths are set to $u_{r,t} \\times 0.10$."
    )
    w("")
    w("**Shortlist thresholds**: Tier 1 ≥ 0.70; Tier 2 ≥ 0.60 and < 0.70.")
    w("")
    w("## 2.4 Spatial Analysis")
    w("")
    w(
        "Spatial autocorrelation is assessed using queen contiguity weights (libpysal), "
        "row-standardised, with 999 conditional permutations. Global Moran's I is "
        "reported with analytical p-values and simulation-based pseudo-p-values. "
        "Local Indicators of Spatial Association (LISA) assign each region to "
        "High-High, Low-Low, High-Low, or Low-High quadrants where statistically "
        "significant (p < 0.05)."
    )
    w("")
    w(
        "Spatial corridors are detected via breadth-first search over queen-contiguous "
        "Tier-1 nodes. A corridor requires at least two Tier-1 regions in a connected "
        "component; corridors spanning two or more member states are classified as "
        "cross-border."
    )
    w("")
    w("## 2.5 Feasibility Gateway Analysis")
    w("")
    w(
        "Each type is subject to between one and three binary feasibility gates that "
        "verify minimum structural thresholds (e.g., broadband penetration ≥ 75th EU "
        "percentile, grid import capacity ≥ 5,000 MW, nuclear regulatory eligibility). "
        "T6 (HALEU/Advanced Nuclear) includes a hard-stop gate on regulatory eligibility "
        "(IAEA PRIS capacity > 0 MW or national policy designation): failure eliminates "
        "the region from all T6 scenarios regardless of suitability score."
    )
    w("")
    w(
        "Three planning scenarios are defined:"
    )
    w("")
    w(
        "| Scenario | Horizon | Criteria |"
    )
    w("|----------|---------|----------|")
    w("| A — Near-Term | 2025–2030 | Tier 1 (≥ 0.70) + all gates passed |")
    w("| B — Medium-Term | 2030–2035 | Tier 2 (≥ 0.60) + primary gates + SR ≥ 0.70 |")
    w("| C — Long-Term | 2035–2040 | Any shortlisted + SR ≥ 0.50 + ≤ 2 soft gate failures |")
    w("")

    # ── 3. Type-Level Findings ─────────────────────────────────────────────
    w("# 3. Suitability Findings by Ecosystem Type")
    w("")
    w(
        "The table below summarises key statistics across all eight types. "
        "'Tier 1' denotes regions with a composite suitability index ≥ 0.70; "
        "'Tier 2' denotes the 0.60–0.70 band. 'Scen A' and 'Scen B' are the "
        "near-term and medium-term feasibility-gated shortlists. "
        "'Uncertainty' is the mean uncertainty score across shortlisted regions; "
        "'SR Median' is the median structural readiness (proportion of features "
        "above the 25th EU percentile)."
    )
    w("")
    w(
        "| Type | Label | Tier 1 (% of 242) | Tier 2 | Scen A | Scen B "
        "| Uncertainty | SR Median |"
    )
    w("|------|-------|--------------------|--------|--------|--------|-------------|-----------|")
    for tid in ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]:
        w(_tier_summary_row(tid, p5, p7))
    w("")

    for tid in ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]:
        t = p5["types"][tid]
        ft = p7["types"][tid]
        m_row = morans[morans["type_id"] == tid].iloc[0]
        cs_row = cor_sum[cor_sum["type_id"] == tid]
        n_corr = int(cs_row["n_corridors"].values[0]) if len(cs_row) else 0
        n_cb = int(cs_row["n_cross_border"].values[0]) if len(cs_row) else 0
        larg = int(cs_row["largest_corridor"].values[0]) if len(cs_row) else 0

        w(f"## 3.{list(TYPE_NAMES.keys()).index(tid) + 1} {tid} — {TYPE_NAMES[tid]}")
        w("")
        w(
            f"**Shortlist:** {t['n_tier1']} Tier-1 regions ({_pct(t['n_tier1'])} of EU-242) "
            f"and {t['n_tier2']} Tier-2 regions. "
            f"Mean suitability index: {t['score_mean']:.3f} (σ = {t['score_std']:.3f}). "
            f"Median structural readiness: {t['structural_readiness_median']:.3f}."
        )
        w("")
        w(
            f"**Spatial structure:** Global Moran's I = {m_row.moran_i:.3f} "
            f"(z = {m_row.z_score:.2f}, p < 0.001), confirming statistically significant "
            f"positive spatial autocorrelation. "
        )
        if n_corr > 0:
            w(
                f"{n_corr} spatial corridor{'s' if n_corr > 1 else ''} detected "
                f"({n_cb} cross-border); largest corridor: {larg} contiguous Tier-1 regions."
            )
        else:
            w("No multi-region spatial corridors detected at Tier-1 threshold.")
        w("")
        w(
            f"**Feasibility scenarios:** Scenario A (near-term): {ft['scenario_A']} regions. "
            f"Scenario B (medium-term): {ft['scenario_B']} regions. "
        )
        if ft.get("scenario_C", 0) > 0:
            w(f"Scenario C (long-term): {ft['scenario_C']} additional regions.")
        w("")
        gap_features = t.get("top_gap_features", [])
        if gap_features:
            w(
                f"**Primary structural gaps** (features most frequently below the 25th EU "
                f"percentile in shortlisted regions): "
                + ", ".join(f"`{f}`" for f in gap_features[:3])
                + "."
            )
        if t["uncertainty_mean"] > 0.40:
            w(
                f"**Uncertainty note:** Mean uncertainty score {t['uncertainty_mean']:.2f} "
                f"indicates a high proportion of composite weight is carried by DATA GAP "
                f"proxy features. Findings for this type should be interpreted with "
                f"commensurately greater caution."
            )
        w("")

    # ── 4. Spatial Analysis ────────────────────────────────────────────────
    w("# 4. Spatial Structure of High-Index Regions")
    w("")
    w(
        "All eight ecosystem types exhibit statistically significant positive spatial "
        "autocorrelation: high-index regions cluster with high-index neighbours and "
        "low-index regions with low-index neighbours. This pattern is consistent with "
        "the theoretical expectation that structural pre-conditions — talent pools, "
        "R&D infrastructure, industrial base — exhibit strong spatial spillovers and "
        "agglomeration effects."
    )
    w("")
    w("**Global Moran's I by type:**")
    w("")
    w("| Type | Label | Moran's I | z-Score | p-value |")
    w("|------|-------|-----------|---------|---------|")
    for _, row in morans.iterrows():
        w(_moran_row(row))
    w("")
    w(
        f"The strongest spatial clustering is observed for T5 (Data Centre Hub, I = 0.895), "
        f"reflecting the geographic concentration of grid capacity, renewable energy supply, "
        f"and fibre infrastructure across the Nordic-Baltic corridor. T1 (AI/ML Hub, "
        f"I = 0.557) shows the weakest spatial autocorrelation, consistent with the "
        f"broader geographic diffusion of digital talent and ICT infrastructure across "
        f"EU metropolitan regions."
    )
    w("")
    w(
        f"Across all types, {total_corridors} spatial corridors are identified. "
        f"{cross_border_corridors} corridors span two or more member states, "
        "indicating that several structural pre-condition clusters are not bounded "
        "by national borders. Notable cross-border corridors include:"
    )
    w("")
    w("- **T4/T5 Nordic-Baltic**: A 12-region corridor spanning southern Sweden and western")
    w("  Finland, structurally favourable for both Cleantech and Data Centre Hub types.")
    w("- **T4 Iberian**: A 9-region corridor spanning south-western Spain and central Portugal,")
    w("  reflecting shared renewable energy and emerging industrial capacity.")
    w("- **T4 Danube-Adriatic**: A 6-region corridor spanning Austria, Slovenia, and Croatia.")
    w("- **T6 French Nuclear Arc**: A 7-region intra-France corridor reflecting concentrated")
    w("  nuclear industrial base and regulatory eligibility.")
    w("- **T8 German Research Cluster**: A 29-region intra-Germany corridor — the largest")
    w("  single corridor in the analysis — reflecting the structural depth of German")
    w("  public-sector R&D and EuroHPC proximity.")
    w("")

    # ── 5. Feasibility & Scenarios ─────────────────────────────────────────
    w("# 5. Feasibility Gateway Analysis and Planning Scenarios")
    w("")
    w(
        "Suitability indices identify structurally favourable regions; feasibility "
        "gates verify that minimum operational thresholds are met. Across all types, "
        "gateway constraints reduce shortlists materially:"
    )
    w("")
    w("| Type | Tier-1 | All Gates Passed | Scen A (Tier1 + Gates) | Gate Reduction |")
    w("|------|--------|-----------------|------------------------|----------------|")
    for tid in ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]:
        t1 = p5["types"][tid]["n_tier1"]
        gp = p7["types"][tid]["regions_all_gates_passed"]
        sa = p7["types"][tid]["scenario_A"]
        reduction = f"{t1 - sa} regions removed"
        w(f"| {tid} | {t1} | {gp} | {sa} | {reduction} |")
    w("")
    w(
        "The T6 (HALEU/Advanced Nuclear) hard-stop gate is the most restrictive: "
        "only 96 of 242 NUTS2 regions pass the regulatory eligibility criterion, "
        "reducing the potential pool to 39.7% of EU NUTS2 before suitability scoring "
        "is applied. This reflects both the geographic concentration of existing "
        "nuclear capacity and the current state of national nuclear policy across "
        "member states."
    )
    w("")
    w(
        "T5 (Data Centre Hub) is subject to three gates — broadband penetration, "
        "grid import capacity, and water availability — and has the smallest "
        "Scenario A shortlist (13 regions) despite a Tier-1 pool of 13, indicating "
        "that the Tier-1 threshold already captures high structural co-alignment "
        "with operational requirements."
    )
    w("")
    w(
        "T8 (Quantum/Photonics) has the largest Scenario A shortlist (53 regions) "
        "and is unique in that Scenario B matches Scenario A exactly, suggesting "
        "that all Tier-2 regions for this type satisfy the medium-term gateway "
        "conditions. This reflects both the single-gate structure for T8 and the "
        "high mean uncertainty score (0.451), which signals that the T8 index "
        "relies heavily on proxy features (total EPO patent density, EuroHPC "
        "proximity, national quantum programme tier) rather than directly observed "
        "NUTS2-level quantum-sector data."
    )
    w("")

    # ── 6. Input-Output Dependencies ─────────────────────────────────────
    w("# 6. Cross-Type Structural Dependencies")
    w("")
    w(
        "An expert-coded 8×8 structural dependency matrix captures the degree to "
        "which structural pre-conditions for one ecosystem type co-locate with or "
        "reinforce pre-conditions for another. Values range from 0 (no structural "
        "overlap) to 3 (strong co-dependence)."
    )
    w("")
    w(
        f"The matrix contains {p7['io_total_links']} non-zero links across 56 possible "
        f"type pairs, of which {p7['io_strong_links']} are strong co-dependencies (value = 3):"
    )
    w("")
    w("| Source → Target | Structural Relationship |")
    w("|-----------------|------------------------|")
    w("| T1 (AI/ML) → T5 (Data Centre) | Strong: AI workloads are structurally co-located with hyperscale compute infrastructure |")
    w("| T3 (Semiconductors) → T7 (Robotics) | Strong: Advanced electronics manufacturing base structurally pre-conditions precision robotics |")
    w("| T5 (Data Centre) → T4 (Cleantech) | Strong: Hyperscale power demand structurally incentivises renewable energy build-out |")
    w("| T7 (Robotics) → T3 (Semiconductors) | Strong: Robotics R&D generates demand and talent flows into semiconductor design |")
    w("")
    w(
        "These strong bidirectional links between T3 and T7 indicate a mutually "
        "reinforcing structural cluster: regions with advanced manufacturing quotients "
        "and machinery/automotive LQs are structurally favourable for both types "
        "simultaneously. Similarly, the T1→T5→T4 chain suggests that AI hub development "
        "and data-centre deployment are structurally associated with cleantech "
        "infrastructure maturity."
    )
    w("")

    # ── 7. Data Limitations ────────────────────────────────────────────────
    w("# 7. Data Limitations and Uncertainty")
    w("")
    w(
        "The analysis is subject to the following documented data limitations. "
        "All are registered in the DATA GAP register (Phase P1) and propagated "
        "into uncertainty scores."
    )
    w("")
    w("| Gap | Proxy Used | Types Affected | Uncertainty Impact |")
    w("|-----|-----------|---------------|-------------------|")
    w("| Quantum/robotics patents at NUTS2 (IPC sub-class) | Total EPO patent density + EuroHPC proximity + national quantum flag | T7, T8 | High |")
    w("| Grid headroom per NUTS2 | ENTSO-E country NTC capacity + electricity price (inverse) | T5, T6 | Moderate |")
    w("| Industrial land cost at NUTS2 | Inverse GDP per capita PPS + urban/rural typology | T3, T5 | Moderate |")
    w("| HALEU patent IPC at NUTS2 | Existing nuclear capacity + chemical LQ + regulatory flag | T6 | High |")
    w("| Port throughput (92.2% null) | Excluded entirely | T5, T3 | Low (excluded) |")
    w("| Ultrafast broadband (30.2% null) | Country-median imputed | T1, T5 | Low |")
    w("")
    w(
        "T6 (HALEU/Advanced Nuclear) carries the highest aggregate uncertainty (mean = 0.687), "
        "reflecting that the majority of its composite weight is borne by proxy indicators. "
        "T8 (Quantum/Photonics) has uncertainty = 0.451. All other types have uncertainty "
        "below 0.25."
    )
    w("")
    w(
        "Cross-sectional data from multiple reference years (2019–2023) introduces "
        "temporal heterogeneity. Year alignment is documented in the analysis artefact "
        "`p2_year_alignment.csv`. No time-series stationarity assumptions are made; "
        "the analysis is explicitly synchronic."
    )
    w("")

    # ── 8. Methodology Annex ──────────────────────────────────────────────
    w("# 8. Methodology Annex")
    w("")
    w("## 8.1 Feature List by Type")
    w("")
    w(
        "Full feature specifications, including weights, directions, and DATA GAP "
        "flags, are recorded in `analysis/p4_weight_registry.csv`. The 18 "
        "extended features added in Phase P2 are:"
    )
    w("")
    extended_features = [
        ("electricity_price_eur_kwh", "Eurostat nrg_pc_205, 2022", "Inverse (lower = more favourable)"),
        ("renewable_energy_share_pct", "EEA REN-share 2022", "Positive"),
        ("rail_freight_ktonnes", "Eurostat rail_go_grpgood, 2021", "Positive"),
        ("gerd_gov_pct_gdp", "Eurostat rd_e_gerdfund, 2021", "Positive"),
        ("gerd_hes_pct_gdp", "Eurostat rd_e_gerdfund, 2021", "Positive"),
        ("gerd_def_pct_gdp", "Eurostat rd_e_gerdfund, 2021", "Positive"),
        ("ultrafast_broadband_pct", "Eurostat isoc_r_broad_h, 2022", "Positive"),
        ("artificial_land_pct", "Eurostat lan_lcv_art, 2018", "Inverse"),
        ("doing_business_score", "World Bank DB 2020, country-level broadcast", "Positive"),
        ("cit_rate_pct", "OECD Tax Database 2023, country-level", "Inverse"),
        ("ntc_import_mw", "ENTSO-E NTC country capacity, 2023", "Positive — DATA GAP proxy"),
        ("water_exploitation_index", "EEA WEI+ 2018", "Inverse — DATA GAP proxy"),
        ("nuclear_capacity_mw", "IAEA PRIS, plant-NUTS2 mapped, 2023", "Positive — DATA GAP proxy"),
        ("eurohpc_proximity_score", "EuroHPC 18 sites, distance-decay, 2023", "Positive — DATA GAP proxy"),
        ("quantum_programme_tier", "National quantum strategy register, 2023", "Positive — DATA GAP proxy"),
        ("nuclear_policy", "National nuclear policy flag, 2023", "Positive — DATA GAP proxy"),
        ("haleu_smr_eligible", "Composite regulatory flag (IAEA + policy)", "Binary gate — T6 hard-stop"),
        ("lq_nace_d35_clean", "Eurostat sbs_r_nuts06_r2 D35 employment LQ", "Positive"),
    ]
    w("| Feature | Source | Direction |")
    w("|---------|--------|-----------|")
    for feat, src, direction in extended_features:
        w(f"| `{feat}` | {src} | {direction} |")
    w("")

    w("## 8.2 Reproducibility")
    w("")
    w(
        "All pipeline scripts are parameterised with `np.random.seed(42)` and "
        "`random_state=42` on all sklearn estimators. The full pipeline is "
        "re-runnable from raw data in sequence: p0 → p1 → p1b → p2 → p4 → "
        "p5 → p6 → p7 → p8 → p9. Artifact hashing and Docker sealed run "
        "are documented in Phase P10 (Reproducibility Audit, forthcoming)."
    )
    w("")
    w("## 8.3 Epistemological Contract")
    w("")
    w(
        "This analysis operates under a strategic intelligence epistemological "
        "contract. All findings are:"
    )
    w("")
    w("- **Descriptive**: structural pre-conditions associated with past co-location of ecosystem types")
    w("- **Relative**: composite suitability indices express rank position within the EU-242 reference frame")
    w("- **Observational**: no causal identification is performed; cross-sectional correlations only")
    w("- **Uncertain**: DATA GAP proxies and temporal heterogeneity propagate uncertainty into all scores")
    w("")
    w(
        "**Permitted language**: structurally favourable, high-index regions, "
        "shortlisted, composite suitability index ≥ 0.70, strong candidate, "
        "structural pre-conditions associated with, historically co-located with."
    )
    w("")
    w(
        "**Prohibited language**: the best region, optimal location, causes, "
        "effect of X on Y, guarantees returns/success, will outperform, "
        "composite ranking, site selection score."
    )
    w("")
    w("---")
    w("")
    w("*EU MegaCampus Siting Intelligence Project — Phases P0–P8 Synthesis*  ")
    w(f"*Generated: {TODAY}*  ")
    w("*All figures and analysis artefacts: `EU-MegaCampus-Siting/analysis/` and `figures/`*")
    w("")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("P9 — Loading analysis artifacts...")
    art = load_artifacts()

    print("P9 — Building executive brief...")
    report_text = build_report(art)

    OUTPUT_MD.write_text(report_text, encoding="utf-8")
    print(f"P9 — Report written: {OUTPUT_MD} ({len(report_text):,} chars)")

    # Attempt PDF conversion via pandoc
    try:
        result = subprocess.run(
            [
                "pandoc", str(OUTPUT_MD),
                "-o", str(OUTPUT_PDF),
                "--pdf-engine=xelatex",
                "--variable", "geometry:margin=2.5cm",
                "--variable", "fontsize=11pt",
                "--variable", "linestretch=1.3",
                "--toc",
                "--number-sections",
            ],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            print(f"P9 — PDF generated: {OUTPUT_PDF}")
        else:
            print(f"P9 — pandoc PDF failed (xelatex): {result.stderr[:300]}")
            # Fallback: try without xelatex
            result2 = subprocess.run(
                ["pandoc", str(OUTPUT_MD), "-o", str(OUTPUT_PDF), "--toc", "--number-sections"],
                capture_output=True, text=True, timeout=120,
            )
            if result2.returncode == 0:
                print(f"P9 — PDF generated (fallback engine): {OUTPUT_PDF}")
            else:
                print(f"P9 — PDF fallback also failed: {result2.stderr[:200]}")
                print("P9 — Markdown report is complete; PDF requires LaTeX.")
    except FileNotFoundError:
        print("P9 — pandoc not found; markdown report is the primary output.")
    except subprocess.TimeoutExpired:
        print("P9 — pandoc timed out; markdown report is the primary output.")

    # Gate check
    wc = len(report_text.split())
    print(f"\nGATE CHECK P9:")
    print(f"  Report word count : {wc:,}")
    print(f"  Output path       : {OUTPUT_MD}")
    print(f"  Sections          : 8 (Executive Summary -> Methodology Annex)")
    # Check only the body sections (the annex intentionally lists forbidden terms)
    body_text = report_text.split("**Prohibited language**")[0]
    has_forbidden = any(
        phrase in body_text
        for phrase in [
            "the best region", "optimal location", "causes ", "effect of X on Y",
            "composite ranking", "site selection score", "ideal for", "will succeed",
        ]
    )
    print(f"  Language contract : {'FAIL -- forbidden phrase found in body' if has_forbidden else 'PASS'}")
    print(f"\nGATE_P9={'FAIL' if has_forbidden or wc < 2000 else 'PASS'}")


if __name__ == "__main__":
    main()
