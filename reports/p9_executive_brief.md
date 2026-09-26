---
title: |
  EU MegaCampus Siting Intelligence
  Executive Intelligence Brief
subtitle: |
  Structural Pre-Conditions for Eight Deep-Tech Ecosystem Types
  Across 242 EU NUTS2 Regions — Phases P0–P8 Synthesis
date: '2026-05-24'
author: EU MegaCampus Siting Intelligence Project
abstract: |
  This brief synthesises structural suitability analysis across 242 EU NUTS2
  regions for eight disruptor ecosystem types: AI/ML hubs, biotechnology
  clusters, semiconductor corridors, cleantech zones, hyperscale data-centre
  hubs, HALEU/advanced-nuclear industrial bases, deep-tech robotics campuses,
  and quantum/photonics research anchors. Composite suitability indices are
  constructed from 84 harmonised structural features. All findings are
  descriptive associations derived from observational cross-sectional data;
  no causal claims are made.
---

# 1. Executive Summary

This intelligence brief presents structural suitability profiles for eight deep-tech disruptor ecosystem types across all 242 NUTS2 regions of the European Union. The analysis integrates 84 harmonised structural features drawn from the EU Innovation Panel Gold layer and 18 purpose-built indicators covering electricity infrastructure, renewable energy, grid capacity, industrial land proxies, regulatory eligibility, and supranational programme participation.

Composite suitability indices are constructed for each type using weighted feature aggregation, per-type min–max rescaling, and propagated uncertainty estimates. Spatial autocorrelation analysis confirms that high-index regions are geographically clustered for all eight types (Global Moran's I range: 0.557–0.895, all p < 0.001). A total of 23 spatial corridors are identified across types, of which 9 span two or more member states, indicating transnational structural alignments that exceed single-region capacity.

Feasibility gateway analysis — encompassing broadband, grid headroom, regulatory eligibility, and industrial base thresholds — confirms that structural suitability is necessary but not sufficient: gateway constraints materially reduce effective shortlists. Under near-term Scenario A (2025–2030), the effective shortlists range from 7 regions (T7 Robotics) to 53 regions (T8 Quantum/Photonics). The HALEU/Advanced Nuclear type (T6) is the most constrained: a hard-stop regulatory-eligibility gate reduces the addressable pool to 96 of 242 regions before suitability scoring is applied.

All findings operate under a strategic intelligence epistemological contract. Indices express relative structural position within the EU-242 reference frame; they do not constitute site-selection recommendations and make no deterministic forecasts about ecosystem emergence or investment outcomes.

# 2. Methodology

## 2.1 Data Architecture and Feature Engineering

The analytical pipeline follows a Bronze → Silver → Gold medallion architecture. The upstream EU Innovation Panel Gold layer (242 NUTS2 × 66 features) provides the structural base, including PCA-derived dimension scores (prosperity, talent, digital infrastructure, innovation cluster), k=4 archetype assignments, and NUTS2-level labour market, R&D, and ICT indicators.

Eighteen additional features were appended at Silver stage from Eurostat, the European Environment Agency, ENTSO-E, the IAEA PRIS reactor database, EuroHPC site records, and national quantum programme registers. The Silver frame contains 242 NUTS2 regions × 84 features at 0.0% null rate after country-median imputation for features with null rates below 10%. The port throughput variable (null rate: 92.2%) is treated as a structural DATA GAP and excluded from scoring; the ultrafast broadband variable (null rate: 30.2%) is imputed from country medians.

## 2.2 Archetype Classification

K-means clustering (k=4, random_state=42) applied to the four PCA dimension scores yields four structural archetypes:

| Archetype | Label | n |
|-----------|-------|---|
| A0 | Established Industrial (n=53) |
| A1 | Catching-Up Peripheral (n=65) |
| A2 | Frontier Innovation (n=43) |
| A3 | Emerging Capacity (n=81) |

The k=4 solution (silhouette = 0.2413) is an intentional interpretability override from the k=2 solution used in the upstream EU Innovation Panel (silhouette = 0.409). Four archetypes are required to map meaningfully onto the eight disruptor-type taxonomy; k=2 conflates structural profiles that differ substantially on the extended feature set. The ARI between k=4 and k=2 is 0.2923, confirming partial but not complete overlap.

## 2.3 Suitability Index Construction

For each of the eight ecosystem types, a composite suitability index is constructed as follows:

1. **Feature selection**: Type-specific feature sets are drawn from the Silver frame. Features are assigned high (H=3), medium (M=2), or low (L=1) weights based on structural relevance documented in the weight registry.
2. **Direction inversion**: Features where lower values indicate greater suitability (e.g., electricity price, water exploitation index) are inverted as 1 − min–max(feature) before scoring.
3. **Weighted composite**: $S_{r,t} = \sum_f w_{f,t} \cdot x_{r,f} / \sum_f w_{f,t}$, where $x_{r,f}$ is the min–max normalised feature value for region $r$ and feature $f$, and $w_{f,t}$ is the type-specific weight.
4. **Per-type rescaling**: Each type's composite scores are min–max rescaled to [0, 1] so that the threshold 0.60 consistently identifies the top 40% of EU NUTS2 regions on that type's structural profile. This is analytically necessary because country-level broadcast features in T4 (Cleantech) compress the raw composite maximum to 0.571, making a common absolute threshold inappropriate.
5. **Uncertainty propagation**: An uncertainty score $u_{r,t}$ is computed as the proportion of type weight carried by DATA GAP proxy features. Confidence half-widths are set to $u_{r,t} \times 0.10$.

**Shortlist thresholds**: Tier 1 ≥ 0.70; Tier 2 ≥ 0.60 and < 0.70.

## 2.4 Spatial Analysis

Spatial autocorrelation is assessed using queen contiguity weights (libpysal), row-standardised, with 999 conditional permutations. Global Moran's I is reported with analytical p-values and simulation-based pseudo-p-values. Local Indicators of Spatial Association (LISA) assign each region to High-High, Low-Low, High-Low, or Low-High quadrants where statistically significant (p < 0.05).

Spatial corridors are detected via breadth-first search over queen-contiguous Tier-1 nodes. A corridor requires at least two Tier-1 regions in a connected component; corridors spanning two or more member states are classified as cross-border.

## 2.5 Feasibility Gateway Analysis

Each type is subject to between one and three binary feasibility gates that verify minimum structural thresholds (e.g., broadband penetration ≥ 75th EU percentile, grid import capacity ≥ 5,000 MW, nuclear regulatory eligibility). T6 (HALEU/Advanced Nuclear) includes a hard-stop gate on regulatory eligibility (IAEA PRIS capacity > 0 MW or national policy designation): failure eliminates the region from all T6 scenarios regardless of suitability score.

Three planning scenarios are defined:

| Scenario | Horizon | Criteria |
|----------|---------|----------|
| A — Near-Term | 2025–2030 | Tier 1 (≥ 0.70) + all gates passed |
| B — Medium-Term | 2030–2035 | Tier 2 (≥ 0.60) + primary gates + SR ≥ 0.70 |
| C — Long-Term | 2035–2040 | Any shortlisted + SR ≥ 0.50 + ≤ 2 soft gate failures |

# 3. Suitability Findings by Ecosystem Type

The table below summarises key statistics across all eight types. 'Tier 1' denotes regions with a composite suitability index ≥ 0.70; 'Tier 2' denotes the 0.60–0.70 band. 'Scen A' and 'Scen B' are the near-term and medium-term feasibility-gated shortlists. 'Uncertainty' is the mean uncertainty score across shortlisted regions; 'SR Median' is the median structural readiness (proportion of features above the 25th EU percentile).

| Type | Label | Tier 1 (% of 242) | Tier 2 | Scen A | Scen B | Uncertainty | SR Median |
|------|-------|--------------------|--------|--------|--------|-------------|-----------|
| T1 | AI/ML Hub | 25 (10%) | 24 | 25 | 24 | 0.08 | 0.800 |
| T2 | Biotech / Life Sciences | 19 (8%) | 27 | 16 | 26 | 0.05 | 0.889 |
| T3 | Semiconductors | 33 (14%) | 42 | 23 | 39 | 0.18 | 0.778 |
| T4 | Cleantech | 52 (21%) | 25 | 45 | 21 | 0.16 | 0.750 |
| T5 | Data Centre Hub | 13 (5%) | 8 | 13 | 8 | 0.09 | 0.750 |
| T6 | Advanced Nuclear | 16 (7%) | 22 | 16 | 21 | 0.69 | 0.833 |
| T7 | Robotics Campus | 8 (3%) | 23 | 7 | 23 | 0.06 | 0.875 |
| T8 | Quantum / Photonics | 53 (22%) | 53 | 53 | 53 | 0.45 | 0.857 |

## 3.1 T1 — AI / Machine Learning Hub

**Shortlist:** 25 Tier-1 regions (10% of EU-242) and 24 Tier-2 regions. Mean suitability index: 0.423 (σ = 0.205). Median structural readiness: 0.800.

**Spatial structure:** Global Moran's I = 0.557 (z = 11.61, p < 0.001), confirming statistically significant positive spatial autocorrelation. 
3 spatial corridors detected (0 cross-border); largest corridor: 5 contiguous Tier-1 regions.

**Feasibility scenarios:** Scenario A (near-term): 25 regions. Scenario B (medium-term): 24 regions. 

**Primary structural gaps** (features most frequently below the 25th EU percentile in shortlisted regions): `electricity_price_eur_kwh`, `cit_rate_pct`, `doing_business_score`.

## 3.2 T2 — Biotechnology / Life Sciences

**Shortlist:** 19 Tier-1 regions (8% of EU-242) and 27 Tier-2 regions. Mean suitability index: 0.432 (σ = 0.187). Median structural readiness: 0.889.

**Spatial structure:** Global Moran's I = 0.606 (z = 12.62, p < 0.001), confirming statistically significant positive spatial autocorrelation. 
2 spatial corridors detected (0 cross-border); largest corridor: 4 contiguous Tier-1 regions.

**Feasibility scenarios:** Scenario A (near-term): 16 regions. Scenario B (medium-term): 26 regions. 
Scenario C (long-term): 4 additional regions.

**Primary structural gaps** (features most frequently below the 25th EU percentile in shortlisted regions): `electricity_price_eur_kwh`, `gerd_hes_pct_gdp`, `doing_business_score`.

## 3.3 T3 — Semiconductors / Advanced Electronics

**Shortlist:** 33 Tier-1 regions (14% of EU-242) and 42 Tier-2 regions. Mean suitability index: 0.492 (σ = 0.187). Median structural readiness: 0.778.

**Spatial structure:** Global Moran's I = 0.619 (z = 12.88, p < 0.001), confirming statistically significant positive spatial autocorrelation. 
5 spatial corridors detected (1 cross-border); largest corridor: 12 contiguous Tier-1 regions.

**Feasibility scenarios:** Scenario A (near-term): 23 regions. Scenario B (medium-term): 39 regions. 
Scenario C (long-term): 13 additional regions.

**Primary structural gaps** (features most frequently below the 25th EU percentile in shortlisted regions): `electricity_price_eur_kwh`, `rail_freight_ktonnes`, `cit_rate_pct`.

## 3.4 T4 — Cleantech / Green Technology

**Shortlist:** 52 Tier-1 regions (21% of EU-242) and 25 Tier-2 regions. Mean suitability index: 0.503 (σ = 0.240). Median structural readiness: 0.750.

**Spatial structure:** Global Moran's I = 0.712 (z = 14.82, p < 0.001), confirming statistically significant positive spatial autocorrelation. 
6 spatial corridors detected (5 cross-border); largest corridor: 12 contiguous Tier-1 regions.

**Feasibility scenarios:** Scenario A (near-term): 45 regions. Scenario B (medium-term): 21 regions. 
Scenario C (long-term): 11 additional regions.

**Primary structural gaps** (features most frequently below the 25th EU percentile in shortlisted regions): `score_talent`, `doing_business_score`, `lq_nace_d35_clean`.

## 3.5 T5 — Hyperscale Data Centre Hub

**Shortlist:** 13 Tier-1 regions (5% of EU-242) and 8 Tier-2 regions. Mean suitability index: 0.378 (σ = 0.204). Median structural readiness: 0.750.

**Spatial structure:** Global Moran's I = 0.894 (z = 18.58, p < 0.001), confirming statistically significant positive spatial autocorrelation. 
1 spatial corridor detected (1 cross-border); largest corridor: 12 contiguous Tier-1 regions.

**Feasibility scenarios:** Scenario A (near-term): 13 regions. Scenario B (medium-term): 8 regions. 

**Primary structural gaps** (features most frequently below the 25th EU percentile in shortlisted regions): `electricity_price_eur_kwh`, `ultrafast_broadband_pct`, `ntc_import_mw`.

## 3.6 T6 — HALEU / Advanced Nuclear Industrial Base

**Shortlist:** 16 Tier-1 regions (7% of EU-242) and 22 Tier-2 regions. Mean suitability index: 0.396 (σ = 0.227). Median structural readiness: 0.833.

**Spatial structure:** Global Moran's I = 0.708 (z = 14.72, p < 0.001), confirming statistically significant positive spatial autocorrelation. 
2 spatial corridors detected (0 cross-border); largest corridor: 7 contiguous Tier-1 regions.

**Feasibility scenarios:** Scenario A (near-term): 16 regions. Scenario B (medium-term): 21 regions. 

**Primary structural gaps** (features most frequently below the 25th EU percentile in shortlisted regions): `lq_nace_c21_m72`, `gerd_def_pct_gdp`, `ntc_import_mw`.
**Uncertainty note:** Mean uncertainty score 0.69 indicates a high proportion of composite weight is carried by DATA GAP proxy features. Findings for this type should be interpreted with commensurately greater caution.

## 3.7 T7 — Deep-Tech Robotics Campus

**Shortlist:** 8 Tier-1 regions (3% of EU-242) and 23 Tier-2 regions. Mean suitability index: 0.402 (σ = 0.171). Median structural readiness: 0.875.

**Spatial structure:** Global Moran's I = 0.633 (z = 13.18, p < 0.001), confirming statistically significant positive spatial autocorrelation. 
1 spatial corridor detected (0 cross-border); largest corridor: 2 contiguous Tier-1 regions.

**Feasibility scenarios:** Scenario A (near-term): 7 regions. Scenario B (medium-term): 23 regions. 
Scenario C (long-term): 1 additional regions.

**Primary structural gaps** (features most frequently below the 25th EU percentile in shortlisted regions): `rail_freight_ktonnes`, `gerd_hes_pct_gdp`, `doing_business_score`.

## 3.8 T8 — Quantum / Photonics Research Anchor

**Shortlist:** 53 Tier-1 regions (22% of EU-242) and 53 Tier-2 regions. Mean suitability index: 0.544 (σ = 0.198). Median structural readiness: 0.857.

**Spatial structure:** Global Moran's I = 0.812 (z = 16.88, p < 0.001), confirming statistically significant positive spatial autocorrelation. 
3 spatial corridors detected (2 cross-border); largest corridor: 29 contiguous Tier-1 regions.

**Feasibility scenarios:** Scenario A (near-term): 53 regions. Scenario B (medium-term): 53 regions. 

**Primary structural gaps** (features most frequently below the 25th EU percentile in shortlisted regions): `ultrafast_broadband_pct`, `score_talent`, `score_cluster`.
**Uncertainty note:** Mean uncertainty score 0.45 indicates a high proportion of composite weight is carried by DATA GAP proxy features. Findings for this type should be interpreted with commensurately greater caution.

# 4. Spatial Structure of High-Index Regions

All eight ecosystem types exhibit statistically significant positive spatial autocorrelation: high-index regions cluster with high-index neighbours and low-index regions with low-index neighbours. This pattern is consistent with the theoretical expectation that structural pre-conditions — talent pools, R&D infrastructure, industrial base — exhibit strong spatial spillovers and agglomeration effects.

**Global Moran's I by type:**

| Type | Label | Moran's I | z-Score | p-value |
|------|-------|-----------|---------|---------|
| T1 | AI/ML Hub | 0.557 | 11.61 | < 0.001 |
| T2 | Biotech / Life Sciences | 0.606 | 12.62 | < 0.001 |
| T3 | Semiconductors | 0.619 | 12.88 | < 0.001 |
| T4 | Cleantech | 0.712 | 14.82 | < 0.001 |
| T5 | Data Centre Hub | 0.894 | 18.58 | < 0.001 |
| T6 | Advanced Nuclear | 0.708 | 14.72 | < 0.001 |
| T7 | Robotics Campus | 0.633 | 13.18 | < 0.001 |
| T8 | Quantum / Photonics | 0.812 | 16.88 | < 0.001 |

The strongest spatial clustering is observed for T5 (Data Centre Hub, I = 0.895), reflecting the geographic concentration of grid capacity, renewable energy supply, and fibre infrastructure across the Nordic-Baltic corridor. T1 (AI/ML Hub, I = 0.557) shows the weakest spatial autocorrelation, consistent with the broader geographic diffusion of digital talent and ICT infrastructure across EU metropolitan regions.

Across all types, 23 spatial corridors are identified. 9 corridors span two or more member states, indicating that several structural pre-condition clusters are not bounded by national borders. Notable cross-border corridors include:

- **T4/T5 Nordic-Baltic**: A 12-region corridor spanning southern Sweden and western
  Finland, structurally favourable for both Cleantech and Data Centre Hub types.
- **T4 Iberian**: A 9-region corridor spanning south-western Spain and central Portugal,
  reflecting shared renewable energy and emerging industrial capacity.
- **T4 Danube-Adriatic**: A 6-region corridor spanning Austria, Slovenia, and Croatia.
- **T6 French Nuclear Arc**: A 7-region intra-France corridor reflecting concentrated
  nuclear industrial base and regulatory eligibility.
- **T8 German Research Cluster**: A 29-region intra-Germany corridor — the largest
  single corridor in the analysis — reflecting the structural depth of German
  public-sector R&D and EuroHPC proximity.

# 5. Feasibility Gateway Analysis and Planning Scenarios

Suitability indices identify structurally favourable regions; feasibility gates verify that minimum operational thresholds are met. Across all types, gateway constraints reduce shortlists materially:

| Type | Tier-1 | All Gates Passed | Scen A (Tier1 + Gates) | Gate Reduction |
|------|--------|-----------------|------------------------|----------------|
| T1 | 25 | 143 | 25 | 0 regions removed |
| T2 | 19 | 161 | 16 | 3 regions removed |
| T3 | 33 | 128 | 23 | 10 regions removed |
| T4 | 52 | 168 | 45 | 7 regions removed |
| T5 | 13 | 92 | 13 | 0 regions removed |
| T6 | 16 | 96 | 16 | 0 regions removed |
| T7 | 8 | 157 | 7 | 1 regions removed |
| T8 | 53 | 208 | 53 | 0 regions removed |

The T6 (HALEU/Advanced Nuclear) hard-stop gate is the most restrictive: only 96 of 242 NUTS2 regions pass the regulatory eligibility criterion, reducing the potential pool to 39.7% of EU NUTS2 before suitability scoring is applied. This reflects both the geographic concentration of existing nuclear capacity and the current state of national nuclear policy across member states.

T5 (Data Centre Hub) is subject to three gates — broadband penetration, grid import capacity, and water availability — and has the smallest Scenario A shortlist (13 regions) despite a Tier-1 pool of 13, indicating that the Tier-1 threshold already captures high structural co-alignment with operational requirements.

T8 (Quantum/Photonics) has the largest Scenario A shortlist (53 regions) and is unique in that Scenario B matches Scenario A exactly, suggesting that all Tier-2 regions for this type satisfy the medium-term gateway conditions. This reflects both the single-gate structure for T8 and the high mean uncertainty score (0.451), which signals that the T8 index relies heavily on proxy features (total EPO patent density, EuroHPC proximity, national quantum programme tier) rather than directly observed NUTS2-level quantum-sector data.

# 6. Cross-Type Structural Dependencies

An expert-coded 8×8 structural dependency matrix captures the degree to which structural pre-conditions for one ecosystem type co-locate with or reinforce pre-conditions for another. Values range from 0 (no structural overlap) to 3 (strong co-dependence).

The matrix contains 14 non-zero links across 56 possible type pairs, of which 4 are strong co-dependencies (value = 3):

| Source → Target | Structural Relationship |
|-----------------|------------------------|
| T1 (AI/ML) → T5 (Data Centre) | Strong: AI workloads are structurally co-located with hyperscale compute infrastructure |
| T3 (Semiconductors) → T7 (Robotics) | Strong: Advanced electronics manufacturing base structurally pre-conditions precision robotics |
| T5 (Data Centre) → T4 (Cleantech) | Strong: Hyperscale power demand structurally incentivises renewable energy build-out |
| T7 (Robotics) → T3 (Semiconductors) | Strong: Robotics R&D generates demand and talent flows into semiconductor design |

These strong bidirectional links between T3 and T7 indicate a mutually reinforcing structural cluster: regions with advanced manufacturing quotients and machinery/automotive LQs are structurally favourable for both types simultaneously. Similarly, the T1→T5→T4 chain suggests that AI hub development and data-centre deployment are structurally associated with cleantech infrastructure maturity.

# 7. Data Limitations and Uncertainty

The analysis is subject to the following documented data limitations. All are registered in the DATA GAP register (Phase P1) and propagated into uncertainty scores.

| Gap | Proxy Used | Types Affected | Uncertainty Impact |
|-----|-----------|---------------|-------------------|
| Quantum/robotics patents at NUTS2 (IPC sub-class) | Total EPO patent density + EuroHPC proximity + national quantum flag | T7, T8 | High |
| Grid headroom per NUTS2 | ENTSO-E country NTC capacity + electricity price (inverse) | T5, T6 | Moderate |
| Industrial land cost at NUTS2 | Inverse GDP per capita PPS + urban/rural typology | T3, T5 | Moderate |
| HALEU patent IPC at NUTS2 | Existing nuclear capacity + chemical LQ + regulatory flag | T6 | High |
| Port throughput (92.2% null) | Excluded entirely | T5, T3 | Low (excluded) |
| Ultrafast broadband (30.2% null) | Country-median imputed | T1, T5 | Low |

T6 (HALEU/Advanced Nuclear) carries the highest aggregate uncertainty (mean = 0.687), reflecting that the majority of its composite weight is borne by proxy indicators. T8 (Quantum/Photonics) has uncertainty = 0.451. All other types have uncertainty below 0.25.

Cross-sectional data from multiple reference years (2019–2023) introduces temporal heterogeneity. Year alignment is documented in the analysis artefact `p2_year_alignment.csv`. No time-series stationarity assumptions are made; the analysis is explicitly synchronic.

# 8. Methodology Annex

## 8.1 Feature List by Type

Full feature specifications, including weights, directions, and DATA GAP flags, are recorded in `analysis/p4_weight_registry.csv`. The 18 extended features added in Phase P2 are:

| Feature | Source | Direction |
|---------|--------|-----------|
| `electricity_price_eur_kwh` | Eurostat nrg_pc_205, 2022 | Inverse (lower = more favourable) |
| `renewable_energy_share_pct` | EEA REN-share 2022 | Positive |
| `rail_freight_ktonnes` | Eurostat rail_go_grpgood, 2021 | Positive |
| `gerd_gov_pct_gdp` | Eurostat rd_e_gerdfund, 2021 | Positive |
| `gerd_hes_pct_gdp` | Eurostat rd_e_gerdfund, 2021 | Positive |
| `gerd_def_pct_gdp` | Eurostat rd_e_gerdfund, 2021 | Positive |
| `ultrafast_broadband_pct` | Eurostat isoc_r_broad_h, 2022 | Positive |
| `artificial_land_pct` | Eurostat lan_lcv_art, 2018 | Inverse |
| `doing_business_score` | World Bank DB 2020, country-level broadcast | Positive |
| `cit_rate_pct` | OECD Tax Database 2023, country-level | Inverse |
| `ntc_import_mw` | ENTSO-E NTC country capacity, 2023 | Positive — DATA GAP proxy |
| `water_exploitation_index` | EEA WEI+ 2018 | Inverse — DATA GAP proxy |
| `nuclear_capacity_mw` | IAEA PRIS, plant-NUTS2 mapped, 2023 | Positive — DATA GAP proxy |
| `eurohpc_proximity_score` | EuroHPC 18 sites, distance-decay, 2023 | Positive — DATA GAP proxy |
| `quantum_programme_tier` | National quantum strategy register, 2023 | Positive — DATA GAP proxy |
| `nuclear_policy` | National nuclear policy flag, 2023 | Positive — DATA GAP proxy |
| `haleu_smr_eligible` | Composite regulatory flag (IAEA + policy) | Binary gate — T6 hard-stop |
| `lq_nace_d35_clean` | Eurostat sbs_r_nuts06_r2 D35 employment LQ | Positive |

## 8.2 Reproducibility

All pipeline scripts are parameterised with `np.random.seed(42)` and `random_state=42` on all sklearn estimators. The full pipeline is re-runnable from raw data in sequence: p0 → p1 → p1b → p2 → p4 → p5 → p6 → p7 → p8 → p9. Artifact hashing and Docker sealed run are documented in Phase P10 (Reproducibility Audit, forthcoming).

## 8.3 Epistemological Contract

This analysis operates under a strategic intelligence epistemological contract. All findings are:

- **Descriptive**: structural pre-conditions associated with past co-location of ecosystem types
- **Relative**: composite suitability indices express rank position within the EU-242 reference frame
- **Observational**: no causal identification is performed; cross-sectional correlations only
- **Uncertain**: DATA GAP proxies and temporal heterogeneity propagate uncertainty into all scores

**Permitted language**: structurally favourable, high-index regions, shortlisted, composite suitability index ≥ 0.70, strong candidate, structural pre-conditions associated with, historically co-located with.

**Prohibited language**: the best region, optimal location, causes, effect of X on Y, guarantees returns/success, will outperform, composite ranking, site selection score.

---

*EU MegaCampus Siting Intelligence Project — Phases P0–P8 Synthesis*  
*Generated: 2026-05-24*  
*All figures and analysis artefacts: `EU-MegaCampus-Siting/analysis/` and `figures/`*
