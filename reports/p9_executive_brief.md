---
title: |
  EU MegaCampus Siting Intelligence
  Executive Intelligence Brief
subtitle: |
  Structural Pre-Conditions for Eight Deep-Tech Ecosystem Types
  Across 242 EU NUTS2 Regions — Phases P0–P9 Synthesis
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
  no causal claims are made. Indices for T6 (Advanced Nuclear) and T8
  (Quantum/Photonics) carry high uncertainty due to reliance on proxy
  indicators and should be treated as structural orientation rather than
  assessment.
---

# 1. Executive Summary

This intelligence brief presents structural suitability profiles for eight deep-tech disruptor ecosystem types across all 242 NUTS2 regions of the European Union. The analysis integrates 84 harmonised structural features drawn from the EU Innovation Panel Gold layer and 18 purpose-built indicators covering electricity infrastructure, renewable energy, grid capacity, industrial land proxies, regulatory eligibility, and supranational programme participation.

Each suitability index is a weighted average of 10–20 structural features — each normalised to a 0–1 scale — where a score of 0.70 or above places a region in the top 30% of all 242 EU NUTS2 regions on that ecosystem type's structural profile. "High-index" is always a relative, not an absolute, designation.

Composite suitability indices are constructed for each type using weighted feature aggregation, per-type min–max rescaling, and propagated uncertainty estimates. Spatial autocorrelation analysis confirms that high-index regions are geographically clustered for all eight types (Global Moran's I range: 0.557–0.895, all p < 0.001). A total of 21 spatial corridors are identified across types, of which 9 span two or more member states, indicating transnational structural co-occurrence at NUTS2 level that may exceed single-region structural depth.

Feasibility gateway analysis — encompassing broadband, grid headroom, regulatory eligibility, and industrial base thresholds — confirms that structural suitability is necessary but not sufficient: gateway constraints materially reduce effective shortlists. Under near-term Scenario A (2025–2030), the effective shortlists range from 7 regions (T7 Robotics) to 53 regions (T8 Quantum/Photonics). The HALEU/Advanced Nuclear type (T6) is the most constrained: a hard-stop regulatory-eligibility gate reduces the addressable pool to 96 of 242 regions before suitability scoring is applied.

All findings operate under a strategic intelligence epistemological contract. Indices express relative structural position within the EU-242 reference frame; they do not constitute site-selection recommendations and make no deterministic forecasts about ecosystem emergence or investment outcomes.

---

# 2. Methodology

## 2.1 Data Architecture and Feature Engineering

The analytical pipeline follows a Bronze → Silver → Gold medallion architecture. The upstream EU Innovation Panel Gold layer (242 NUTS2 × 66 features) provides the structural base, including PCA-derived dimension scores (prosperity, talent, digital infrastructure, innovation cluster), k=4 archetype assignments, and NUTS2-level labour market, R&D, and ICT indicators.

Eighteen additional features were appended at Silver stage from Eurostat, the European Environment Agency, ENTSO-E, the IAEA PRIS reactor database, EuroHPC site records, and national quantum programme registers. The Silver frame contains 242 NUTS2 regions × 84 features at 0.0% null rate after country-median imputation for features with null rates below 10%. The port throughput variable (null rate: 92.2%) is treated as a structural DATA GAP and excluded from scoring; the ultrafast broadband variable (null rate: 30.2%) is imputed from country medians.

Five features are available only at country level and are broadcast identically to all NUTS2 regions within a country: `doing_business_score`, `cit_rate_pct`, `renewable_energy_share_pct`, `ntc_import_mw`, and the GERD-by-funding-sector variables. These eliminate all within-country NUTS2 variance on those dimensions and are documented as `variance_scope: country` in the weight registry. For T4 (Cleantech), which is highly weighted on renewable energy share and government R&D, this is a material limitation: the T4 index cannot structurally differentiate NUTS2 regions within the same country on its two highest-weighted features.

## 2.2 Archetype Classification

K-means clustering (k=4, random_state=42) applied to the four PCA dimension scores yields four structural archetypes:

| Archetype | Label | n |
|-----------|-------|---|
| A0 | Established Industrial | 53 |
| A1 | Catching-Up Peripheral | 65 |
| A2 | Frontier Innovation | 43 |
| A3 | Emerging Capacity | 81 |

The k=4 solution (silhouette = 0.2413) is an intentional interpretability override from the k=2 solution used in the upstream EU Innovation Panel (silhouette = 0.409). Four archetypes are required to map meaningfully onto the eight disruptor-type taxonomy; k=2 conflates structural profiles that differ substantially on the extended feature set. The ARI between k=4 and k=2 is 0.2923, confirming partial but not complete overlap. The low silhouette coefficient (below the conventional 0.35 threshold) means archetype boundaries are interpretive rather than statistically crisp; archetype labels should be read as structural orientation, not hard classification.

---

> **Popper Box: Structural Hypotheses Under Test**
>
> Three conjectures embedded in this analysis carry genuine empirical risk and should be treated as provisional rather than established findings.
>
> *Conjecture 1 — Archetype validity:* The k=4 structural archetypes meaningfully differentiate deep-tech hosting propensity beyond what GDP per capita alone captures. Falsifiable: if future Regional Innovation Scoreboard vintages show no systematic RIS score gradient across A0–A3 after controlling for GDP, the archetype layer adds no explanatory value.
>
> *Conjecture 2 — Corridor structural advantage:* Regions embedded in multi-member queen-contiguous Tier-1 corridors are structurally better placed than isolated Tier-1 regions for complex, supply-chain-intensive ecosystem types. Falsifiable: if structural readiness scores of isolated Tier-1 regions are statistically indistinguishable from corridor members of the same archetype and type (Mann-Whitney p > 0.10), the corridor construct is a geometric artefact rather than a structural signal.
>
> *Conjecture 3 — Index predictive association:* Tier-1 shortlisted regions will show above-average Horizon Europe EIC grant success rates in 2025–2030 award data, relative to matched non-Tier-1 regions of the same archetype. This is the strongest out-of-sample test available for index validity; a null result would require reweighting or reconceptualising the composite.
>
> All three are hypotheses about structural association, not causal mechanisms. No claim is made that corridor membership *causes* ecosystem formation, or that a high index *causes* grant success.

---

## 2.3 Suitability Index Construction

For each of the eight ecosystem types, a composite suitability index is constructed as follows:

1. **Feature selection**: Type-specific feature sets are drawn from the Silver frame. Features are assigned high (H=3), medium (M=2), or low (L=1) weights based on structural relevance documented in the weight registry, tagged with `uncertainty_class` and `variance_scope`.
2. **Direction inversion**: Features where lower values indicate greater suitability (e.g., electricity price, water exploitation index) are inverted as 1 − min–max(feature) before scoring.
3. **Weighted composite**: $S_{r,t} = \sum_f w_{f,t} \cdot x_{r,f} / \sum_f w_{f,t}$, where $x_{r,f}$ is the min–max normalised feature value for region $r$ and feature $f$, and $w_{f,t}$ is the type-specific weight.
4. **Per-type rescaling**: Each type's composite scores are min–max rescaled to [0, 1] so that the threshold 0.60 consistently identifies the top 40% of EU NUTS2 regions on that type's structural profile. This is analytically necessary because country-level broadcast features in T4 (Cleantech) compress the raw composite maximum to 0.571, making a common absolute threshold inappropriate across types.
5. **Uncertainty propagation**: An uncertainty score $u_{r,t}$ is computed as the proportion of type weight carried by DATA GAP proxy features. Confidence half-widths are set to $u_{r,t} \times 0.10$ (design parameter, not a statistical estimate — see Deming Box).

**Shortlist thresholds**: Tier 1 ≥ 0.70; Tier 2 ≥ 0.60 and < 0.70.

---

> **Deming Box: Imputation, Variance, and What "Uncertainty" Actually Means**
>
> Three distinct sources of imprecision operate in this analysis and affect different types differently.
>
> *1 — DATA GAP proxy variance.* Some features that would ideally be observed at NUTS2 level (e.g., grid headroom, HALEU patent density) do not exist in any public dataset. They are replaced by country-level or distance-decay proxies. Proxy variance is real structural uncertainty: the proxy may not rank regions the same way the true variable would.
>
> *2 — Imputation-induced variance compression.* Ultrafast broadband (30.2% null) is imputed from country medians. This sets all imputed regions in a country to the same value, suppressing within-country variance on exactly the dimension — digital infrastructure — on which metropolitan vs. rural NUTS2 regions are most likely to differ.
>
> *3 — Country-level broadcast elimination of variance.* Five features are available only at country level and are broadcast identically to all NUTS2 regions within a country. For T4 (Cleantech), this means the index cannot structurally distinguish between regions within the same country on its two top-weighted features. The T4 shortlist (52 Tier-1 regions, 21% of EU-242) is larger than all other types partly because of this inflation.
>
> *Verbal example — T6, a hypothetical French region.* Suppose a region has a T6 composite of 0.78. Its uncertainty score is 0.69, meaning 69% of T6's composite weight is borne by proxy or DATA GAP features (nuclear capacity mapping, HALEU regulatory flag, chemical LQ proxy). Confidence half-width: 0.69 × 0.10 = 0.069. The reported index is 0.78 ± 0.07. A decision made on the difference between 0.78 and 0.71 — both technically Tier-1 — is a decision made within the uncertainty band; the two regions are structurally indistinguishable given the data quality.
>
> *Sensitivity note.* The half-width multiplier of 0.10 is a design choice, not a propagation model. Under a conservative multiplier of 0.15, all T6 bands widen to ±0.10, rendering borderline-Tier-1 T6 regions formally ambiguous. Sensitivity runs at 0.10, 0.12, and 0.15 are planned for Phase P4 revision.

---

## 2.4 Spatial Analysis

Spatial autocorrelation is assessed using queen contiguity weights (libpysal), row-standardised, with 999 conditional permutations. Global Moran's I is reported with analytical p-values and simulation-based pseudo-p-values. Local Indicators of Spatial Association (LISA) assign each region to High-High, Low-Low, High-Low, or Low-High quadrants where statistically significant (p < 0.05).

Spatial corridors are detected via breadth-first search over queen-contiguous Tier-1 nodes. A corridor requires at least two Tier-1 regions in a connected component; corridors spanning two or more member states are classified as cross-border. Corridor membership does not imply functional integration, shared governance, or industrial coherence — only structural co-occurrence at NUTS2 contiguity.

## 2.5 Feasibility Gateway Analysis

Each type is subject to between one and three binary feasibility gates that verify minimum structural thresholds (e.g., broadband penetration ≥ 75th EU percentile, grid import capacity ≥ 5,000 MW, nuclear regulatory eligibility). T6 (HALEU/Advanced Nuclear) includes a hard-stop gate on regulatory eligibility (IAEA PRIS capacity > 0 MW or national policy designation): failure eliminates the region from all T6 scenarios regardless of suitability score.

Three planning scenarios are defined:

| Scenario | Horizon | Criteria |
|----------|---------|----------|
| A — Near-Term | 2025–2030 | Tier 1 (≥ 0.70) + all gates passed |
| B — Medium-Term | 2030–2035 | Tier 2 (≥ 0.60) + primary gates + SR ≥ 0.70 |
| C — Long-Term | 2035–2040 | Any shortlisted + SR ≥ 0.50 + ≤ 2 soft gate failures |

## 2.6 Scale and Heterogeneity: What NUTS2 Indices Cannot Resolve

NUTS2 regions are administrative units averaging 800,000–2,000,000 inhabitants and frequently spanning heterogeneous urban cores, peri-urban zones, and rural hinterlands within the same statistical boundary. A composite suitability index assigned to a NUTS2 region is a weighted average across all its constituent sub-areas; it is not a property of any specific site, district, or campus within that region.

Several consequences follow directly:

- **Within-region divergence can be large.** A high-index NUTS2 region may contain both a globally competitive metropolitan innovation district and a structurally lagging rural periphery. The index reflects neither; it reflects their average. A mid-ranking NUTS2 region may contain a single exceptional sub-regional industrial cluster that substantially outperforms the NUTS2 mean.
- **Corridor coherence is geographic, not functional.** The 12-region Nordic-Baltic T4/T5 corridor spans multiple distinct labour markets, grid operators, and national regulatory regimes. Its spatial continuity at NUTS2 level does not imply that a coherent cross-regional industrial ecosystem exists or is feasible; it indicates shared structural pre-conditions at aggregate scale.
- **NUTS2 is a first-pass filter, not a micro-siting instrument.** The appropriate use of these indices is to narrow the analytical field from 242 regions to a manageable Tier-1 or Tier-2 shortlist, after which NUTS3, Functional Urban Area, or site-specific analysis is required before any operational conclusions can be drawn.

*Potential Phase P12 (indicative, not committed):* For 3–5 high-priority Tier-1 NUTS2 regions per type, a down-scaling exercise to NUTS3 using Eurostat NUTS3 labour market, business register, and land use datasets — supplemented by Urban Audit indicators for functional urban areas — would substantially increase resolution. This is particularly warranted for T5 (Data Centre), where site-specific grid connection capacity and water rights dominate feasibility far more than NUTS2 aggregates can capture.

---

# 3. Suitability Findings by Ecosystem Type

**Exhibit 1: Structural Suitability Profile — All Eight Ecosystem Types.** Each index is a weighted average of 10–20 structural features normalised to [0, 1]; Tier 1 (≥ 0.70) denotes the top 30% of EU-242 on that type's relative structural profile. Uncertainty class: Low < 0.15, Medium 0.15–0.40, High > 0.40.

| Type | Ecosystem Label | Primary Structural Axes | Key Feasibility Gates | Tier 1 (n / %) | Moran's I | Cross-border Corridors | Scen A / B | Uncertainty | Key DATA GAP |
|------|----------------|------------------------|-----------------------|---------------|-----------|------------------------|------------|-------------|-------------|
| **T1** | AI / ML Hub | HRST density · KIS LQ (`lq_nace_j62j63`) · BERD | Broadband ≥ 75th pct | 25 / 10% | 0.557 | 0 of 3 | 25 / 24 | Low (0.08) | None material |
| **T2** | Biotech / Life Sciences | KIS hi-tech LQ (`lq_nace_c21_m72`) · university GERD · EPO patents | Broadband ≥ 75th pct | 19 / 8% | 0.606 | 0 of 2 | 16 / 26 | Low (0.05) | None material |
| **T3** | Semiconductors | Advanced mfg LQ (`lq_nace_c26`) · industrial employment · energy cost | Broadband · energy price | 17 / 7% | 0.614 | 1 of 3 | 13 / 18 | **Medium (0.25)** | Industrial land cost (proxy) |
| **T4** | Cleantech | Renewable share† · energy LQ (`lq_nace_d35_clean`) · gov GERD† | Renewable share ≥ median | 52 / 21% | 0.712 | 5 of 6 | 45 / 21 | Medium (0.16) | Country-broadcast inflation† |
| **T5** | Data Centre Hub | Grid capacity (NTC) · electricity price · broadband | Broadband · grid ≥ 5 GW · WEI | 13 / 5% | **0.895** | 1 of 1 | 13 / 8 | Low (0.09) | Grid headroom (country proxy) |
| **T6** | HALEU / Nuclear | Nuclear capacity (MW) · chemical LQ · regulatory flag | **Hard-stop: eligibility** | 16 / 7% | 0.708 | 0 of 2 | 16 / 21 | **HIGH (0.69)** | HALEU patents; grid headroom |
| **T7** | Robotics Campus | Machinery/auto LQ · university R&D · CORDIS EIC funding | Broadband · industrial LQ | 8 / 3% | 0.637 | 0 of 1 | 7 / 24 | Low (0.05) | Quantum/robotics patents (proxy) |
| **T8** | Quantum / Photonics | EPO patents · EuroHPC proximity · national quantum tier | National programme flag | 53 / 22% | 0.812 | 2 of 3 | 53 / 53 | **HIGH (0.45)** | NUTS2 quantum patents (proxy) |

*† T4 has three country-level broadcast features among its top-weighted variables. The index cannot distinguish NUTS2 regions within the same country on those dimensions. The 52-region Tier-1 count should be interpreted as a national-pattern signal, not a fine-grained NUTS2 discrimination.*

*‡ T8's identical Scenario A and B counts (53/53) indicate that all T8 Tier-2 regions satisfy medium-term gateway conditions. Given T8's high uncertainty (0.45) and single-gate structure, this result reflects index architecture as much as genuine structural differentiation.*

> **T6 Uncertainty Note.** Mean uncertainty score 0.69 — the highest of any type. The majority of T6's composite weight is borne by proxy indicators (nuclear capacity mapping, HALEU regulatory flag, chemical LQ as a nuclear-sector proxy). T6 findings indicate structural orientation consistent with advanced nuclear hosting; they do not constitute an assessment of nuclear readiness, regulatory compliance, or site feasibility. The hard-stop eligibility gate (96 of 242 regions pass) is the most reliable T6 output; the index scores among those 96 should be treated with the stated uncertainty bounds.

> **T8 Uncertainty Note.** Mean uncertainty score 0.45. T8 relies on total EPO patent density (not quantum-IPC-specific), EuroHPC site proximity (distance-decay), and a national quantum programme tier (country-level broadcast). No directly observed NUTS2-level quantum sector data exists in any public dataset at the time of analysis. The 53-region shortlist should be read as "regions with structural pre-conditions broadly associated with science-intensive research and digital infrastructure" rather than as a quantum sector capability map.

---

# 4. Spatial Structure of High-Index Regions

All eight ecosystem types exhibit statistically significant positive spatial autocorrelation: high-index regions cluster with high-index neighbours and low-index regions with low-index neighbours. This pattern is consistent with the theoretical expectation that structural pre-conditions — talent pools, R&D infrastructure, industrial base — exhibit spatial persistence and agglomeration effects that do not respect NUTS2 administrative boundaries.

**Global Moran's I by type:**

| Type | Label | Moran's I | z-Score | p-value |
|------|-------|-----------|---------|---------|
| T1 | AI/ML Hub | 0.557 | 11.61 | < 0.001 |
| T2 | Biotech / Life Sciences | 0.606 | 12.62 | < 0.001 |
| T3 | Semiconductors | 0.614 | 12.78 | < 0.001 |
| T4 | Cleantech | 0.712 | 14.82 | < 0.001 |
| T5 | Data Centre Hub | 0.895 | 18.58 | < 0.001 |
| T6 | Advanced Nuclear | 0.708 | 14.72 | < 0.001 |
| T7 | Robotics Campus | 0.637 | 13.27 | < 0.001 |
| T8 | Quantum / Photonics | 0.812 | 16.88 | < 0.001 |

T5 (I = 0.895) and T8 (I = 0.812) show the strongest spatial clustering; T1 (I = 0.557) the weakest, consistent with the broader geographic diffusion of digital talent and ICT infrastructure across EU metropolitan regions.

Across all types, 21 spatial corridors are identified. 9 corridors span two or more member states, indicating structural co-occurrence that is not bounded by national borders. Notable cross-border corridors (with corridor IDs from `p6_corridors.csv`):

- **T5-C00 Nordic-Baltic (FI+SE):** 12-region corridor spanning southern Sweden and western Finland, structurally favourable for both Cleantech (T4-C02) and Data Centre Hub types — the only corridor appearing in both.
- **T4-C06 Iberian (ES+PT):** 9-region corridor spanning south-western Spain and central Portugal, reflecting shared renewable energy profiles and emerging cleantech industrial capacity.
- **T4-C01 Danube-Adriatic (AT+HR+SI):** 6-region corridor spanning Austria, Slovenia, and Croatia.
- **T6-C00 French Nuclear Arc (FR):** 7-region intra-France corridor reflecting concentrated nuclear industrial base and regulatory eligibility.
- **T8-C00 German Research Cluster (DE):** 29-region intra-Germany corridor — the largest single corridor in the analysis — reflecting the structural depth of German public-sector R&D and EuroHPC proximity.

**Caveat on corridor coherence:** Corridor identification is based on queen contiguity among Tier-1 regions; it does not imply functional integration, shared governance, or a coherent industrial strategy across the corridor. The Nordic-Baltic 12-region corridor, for example, spans multiple distinct grid operators, labour markets, and national investment regimes. Structural co-occurrence at NUTS2 contiguity is a necessary but not sufficient condition for transnational ecosystem viability.

---

# 5. Feasibility Gateway Analysis and Planning Scenarios

Suitability indices identify structurally favourable regions; feasibility gates verify that minimum operational thresholds are met. Across all types, gateway constraints reduce shortlists materially:

| Type | Tier-1 | All Gates Passed | Scen A (Tier1 + Gates) | Gate Reduction |
|------|--------|-----------------|------------------------|----------------|
| T1 | 25 | 143 | 25 | 0 regions removed |
| T2 | 19 | 161 | 16 | 3 regions removed |
| T3 | 17 | 128 | 13 | 4 regions removed |
| T4 | 52 | 168 | 45 | 7 regions removed |
| T5 | 13 | 92 | 13 | 0 regions removed |
| T6 | 16 | 96 | 16 | 0 regions removed |
| T7 | 8 | 157 | 7 | 1 region removed |
| T8 | 53 | 208 | 53 | 0 regions removed |

The T6 (HALEU/Advanced Nuclear) hard-stop gate is the most structurally restrictive: only 96 of 242 NUTS2 regions pass the regulatory eligibility criterion, reducing the potential pool to 39.7% of EU NUTS2 before suitability scoring is applied. This reflects the geographic concentration of existing nuclear capacity and the current state of national nuclear policy across member states.

T5 (Data Centre Hub) is subject to three gates — broadband penetration, grid import capacity, and water availability — and has the smallest absolute Scenario A shortlist (13 regions), with zero attrition from Tier-1 to Scenario A: the Tier-1 threshold already captures high structural co-alignment with gateway conditions.

T8 (Quantum/Photonics) has the largest Scenario A shortlist (53 regions), identical to Scenario B. This reflects both the single-gate structure for T8 and the high mean uncertainty score (0.45), which signals that the T8 index is built primarily from proxy features rather than directly observed NUTS2-level quantum-sector data. The 53-region figure should be read alongside the T8 uncertainty note in Section 3.

---

# 6. Cross-Type Structural Dependencies

An expert-coded 8×8 structural dependency matrix captures the degree to which structural pre-conditions for one ecosystem type co-locate with or are consistent with pre-conditions for another. Values range from 0 (no structural overlap) to 3 (strong co-dependence). **This matrix represents a structural hypothesis, not an empirical finding.** Empirical validation against observed co-location patterns and pairwise Spearman rank correlations across 242 regions is required (planned Phase P7 extension) before these links should be treated as confirmed structural relationships.

The matrix contains 14 non-zero links across 56 possible type pairs, of which 4 are strong co-dependencies (value = 3):

| Source → Target | Structural Relationship |
|-----------------|------------------------|
| T1 (AI/ML) → T5 (Data Centre) | Strong: AI compute workloads are structurally co-located with hyperscale infrastructure — consistent with observed geographic overlap of AI cluster indicators and data centre density |
| T3 (Semiconductors) → T7 (Robotics) | Strong: Advanced electronics manufacturing base is structurally associated with precision robotics capacity via shared advanced materials and metrology infrastructure |
| T5 (Data Centre) → T4 (Cleantech) | Strong: Hyperscale power demand is structurally associated with renewable energy infrastructure maturity in co-located regions |
| T7 (Robotics) → T3 (Semiconductors) | Strong: Robotics R&D and precision manufacturing are structurally associated with semiconductor design capacity via shared HRST and industrial base |

The bidirectional T3↔T7 link is an expert-coded hypothesis that requires empirical validation. If Spearman rank correlation between T3 and T7 composite indices across 242 regions is below ρ = 0.60 in either direction, the symmetry claim requires revision.

The T1→T5→T4 structural chain — where AI hub development and data-centre deployment are associated with cleantech infrastructure maturity — is similarly conjectural: it is consistent with observable co-location patterns in specific member states but has not been formally tested against the full EU-242 data at the type-index level.

---

# 7. Data Limitations and Uncertainty

The analysis is subject to the following documented data limitations. All are registered in the DATA GAP register (Phase P1) and propagated into uncertainty scores.

| Gap | Proxy Used | Types Affected | Uncertainty Impact |
|-----|-----------|---------------|-------------------|
| Quantum/robotics patents at NUTS2 (IPC sub-class) | Total EPO patent density + EuroHPC proximity + national quantum flag | T7, T8 | **High** |
| Grid headroom per NUTS2 | ENTSO-E country NTC capacity + electricity price (inverse) | T5, T6 | Moderate |
| Industrial land cost at NUTS2 | Inverse GDP per capita PPS + urban/rural typology | T3, T5 | Moderate |
| HALEU patent IPC at NUTS2 | Existing nuclear capacity + chemical LQ + regulatory flag | T6 | **High** |
| Port throughput (92.2% null) | Excluded entirely | T5, T3 | Low (excluded) |
| Ultrafast broadband (30.2% null) | Country-median imputed — suppresses within-country variance | T1, T5 | Low-Moderate |
| Country-level broadcast features (5 variables) | Broadcast to all NUTS2 within country — eliminates within-country variance | T4 (primary), T2, T3, T6 | **Medium — inflates apparent T4 shortlist** |
| Institutional / governance capacity | Not proxied — unobserved dimension | All eight types | **Unquantified — see Phase P11b** |

T6 (HALEU/Advanced Nuclear) carries the highest aggregate uncertainty (mean = 0.687), reflecting that the majority of its composite weight is borne by proxy indicators. T8 (Quantum/Photonics) has uncertainty = 0.451. All other types have uncertainty below 0.25.

Cross-sectional data from multiple reference years (2019–2023) introduces temporal heterogeneity. Year alignment is documented in `analysis/p2_year_alignment.csv`. No time-series stationarity assumptions are made; the analysis is explicitly synchronic.

**Unobserved dimension — institutional and governance capacity.** The Silver frame captures structural pre-conditions legible in administrative statistics. It does not capture the institutional capacity of regional actors to identify, coordinate around, and mobilise those structural assets. Two structurally similar regions can have very different practical capacities to host a deep-tech campus depending on governance arrangements — the presence or absence of cluster organisations, regional development agencies, fiscal autonomy, or active Smart Specialisation strategies — that do not appear in any feature currently in the dataset. This constitutes an unobserved source of residual variance across all eight types and is the primary target of planned Phase P11b.

---

# 8. Methodology Annex

## 8.1 Feature List by Type

Full feature specifications, including weights, directions, `uncertainty_class`, and `variance_scope`, are recorded in `analysis/p4_weight_registry.csv`. The 18 extended features added in Phase P2 are:

| Feature | Source | Direction | uncertainty_class | variance_scope |
|---------|--------|-----------|-------------------|----------------|
| `electricity_price_eur_kwh` | Eurostat nrg_pc_205, 2022 | Inverse | direct_country_broadcast | country |
| `renewable_energy_share_pct` | EEA REN-share 2022 | Positive | direct_country_broadcast | country |
| `rail_freight_ktonnes` | Eurostat rail_go_grpgood, 2021 | Positive | direct_NUTS2 | NUTS2 |
| `gerd_gov_pct_gdp` | Eurostat rd_e_gerdfund, 2021 | Positive | direct_country_broadcast | country |
| `gerd_hes_pct_gdp` | Eurostat rd_e_gerdfund, 2021 | Positive | direct_country_broadcast | country |
| `gerd_def_pct_gdp` | Eurostat rd_e_gerdfund, 2021 | Positive | direct_country_broadcast | country |
| `ultrafast_broadband_pct` | Eurostat isoc_r_broad_h, 2022 | Positive | imputed_country_median | NUTS2 (partial) |
| `artificial_land_pct` | Eurostat lan_lcv_art, 2018 | Inverse | direct_NUTS2 | NUTS2 |
| `doing_business_score` | World Bank DB 2020 | Positive | direct_country_broadcast | country |
| `cit_rate_pct` | OECD Tax Database 2023 | Inverse | direct_country_broadcast | country |
| `ntc_import_mw` | ENTSO-E NTC country capacity, 2023 | Positive | DATA_GAP_proxy | country |
| `water_exploitation_index` | EEA WEI+ 2018 | Inverse | DATA_GAP_proxy | NUTS2 (approx) |
| `nuclear_capacity_mw` | IAEA PRIS, plant-NUTS2 mapped, 2023 | Positive | DATA_GAP_proxy | NUTS2 (mapped) |
| `eurohpc_proximity_score` | EuroHPC 18 sites, distance-decay, 2023 | Positive | DATA_GAP_proxy | NUTS2 (derived) |
| `quantum_programme_tier` | National quantum strategy register, 2023 | Positive | DATA_GAP_proxy | country |
| `nuclear_policy` | National nuclear policy flag, 2023 | Positive | DATA_GAP_proxy | country |
| `haleu_smr_eligible` | Composite regulatory flag (IAEA + policy) | Binary gate | DATA_GAP_proxy | country/NUTS2 |
| `lq_nace_d35_clean` | Eurostat sbs_r_nuts06_r2 D35 employment LQ | Positive | direct_NUTS2 | NUTS2 |

## 8.2 Quality Control and Reproducibility

- **Seed policy:** `np.random.seed(42)` at module top level; `random_state=42` on all sklearn estimators. Seeds are logged via structlog pipeline step context managers.
- **Bronze immutability:** Raw source files in `data/bronze/` are never modified. Every transformation is applied at Silver or Gold stage with explicit schema contracts and null-rate audits recorded in `analysis/p2_ingestion_report.json`.
- **Full pipeline re-runability:** The complete pipeline (p0 → p1 → p1b → p2 → p4 → p5 → p6 → p7 → p8 → p9) can be re-executed from raw Bronze files in a single sequential run without manual intervention. Artifact hashing and a Docker sealed run are documented in Phase P10 (forthcoming).
- **"Gold standard" reproducibility** — operationally defined: *an external analyst with access to the listed public sources (Eurostat, EEA, ENTSO-E, IAEA PRIS, EuroHPC site list, World Bank DB, OECD Tax Database) and no access to our code can reproduce the Tier-1 shortlists with Spearman rank correlation ρ > 0.90 per type against our published Gold indices.* This criterion has not yet been externally validated; it is the target for Phase P10.
- **Known limits of reproducibility:** Country-median imputation for ultrafast broadband and the distance-decay scoring of EuroHPC proximity introduce choices not fully constrained by public data. These are declared in `p4_weight_registry.csv` and must be disclosed in any citation of this work.

## 8.3 Epistemological Contract

This analysis operates under a strategic intelligence epistemological contract. All findings are:

- **Descriptive**: structural pre-conditions associated with past co-location of ecosystem types
- **Relative**: composite suitability indices express rank position within the EU-242 reference frame
- **Observational**: no causal identification is performed; cross-sectional correlations only
- **Uncertain**: DATA GAP proxies, country-level broadcasts, and temporal heterogeneity propagate uncertainty into all scores
- **Structurally incomplete**: the institutional and governance capacity of regional actors to mobilise structural assets is not captured in the current feature set and constitutes an unobserved source of residual variance across all types

**Permitted language**: structurally favourable, high-index regions, shortlisted, composite suitability index ≥ 0.70, strong candidate, structural pre-conditions associated with, historically co-located with, is structurally associated with.

**Prohibited language**: the best region, optimal location, causes, effect of X on Y, guarantees returns/success, will outperform, composite ranking, site selection score, structurally incentivises (implies direction), reflects (when explaining I-O links — use "is consistent with").

---

*EU MegaCampus Siting Intelligence Project — Phases P0–P9 Synthesis*
*Socratic Panel Review Applied: Popper, Deming, Hayek, Shannon, Tufte, Jacobs, Ostrom*
*Generated: 2026-05-24*
*All figures and analysis artefacts: `EU-MegaCampus-Siting/analysis/` and `figures/`*
