# CLAUDE.md — EU MegaCampus Siting Intelligence

## Project Purpose
Quantitative site-intelligence analysis identifying NUTS2 regions whose structural
characteristics are associated with hosting eight disruptor ecosystem types:
AI/ML hubs, biotechnology clusters, semiconductor corridors, cleantech zones,
hyperscale data-centre hubs, HALEU/advanced-nuclear industrial bases, deep-tech
robotics campuses, and quantum/photonics research anchors.

This project operates under a **strategic intelligence** epistemological contract.
Composite suitability indices, shortlists, and structural threshold analyses are
explicitly in scope. Descriptive associations, not causal claims.

## Relationship to EU-Innovation-Panel
This project consumes the EU-Innovation-Panel Gold layer as its upstream base:
`../EU-Innovation-Panel/data/gold/region_profiles_gold.parquet` (242 × 66)

The Innovation Panel completed Phases P0–P9 with all gates PASS:
- Silver: 242 regions × 18 features, 100% NUTS2 coverage
- PCA dimension scores: prosperity, talent, digital_infra, innovation_cluster
- k=2 archetype solution (silhouette=0.409) — documented in Innovation Panel
- Full manuscript at `../EU-Innovation-Panel/reports/p9_manuscript.md`

This project re-clusters with **k=4** (intentional interpretability override from k=2).
The k=4 solution will have silhouette < 0.35 threshold used in the base study.
This is an explicit, documented decision: four archetypes are required for the
disruptor-type taxonomy to map meaningfully (k=2 conflates too many structural profiles
that differ substantially on the extended feature set). Document silhouette and ARI in
PHASE 2 gate check; label the override explicitly.

## Eight Disruptor Ecosystem Types
| ID | Label | Primary Structural Indicators |
|----|-------|-------------------------------|
| T1 | AI / Machine Learning Hub | HRST, KIS LQ, BERD, tertiary attainment, broadband |
| T2 | Biotechnology / Life Sciences | KIS hi-tech LQ (M72/C21 proxy), university R&D, HRST, EPO patents |
| T3 | Semiconductors / Advanced Electronics | Advanced mfg LQ (C26), industrial employment, energy cost, transport |
| T4 | Cleantech / Green Technology | Renewable energy share, energy LQ (D35 transition), GERD, BERD |
| T5 | Hyperscale Data Centre Hub | Grid headroom proxy, electricity cost (inverse), broadband, water, land |
| T6 | HALEU / Advanced Nuclear Industrial Base | Nuclear capacity proximity, chemical LQ, regulatory flag, grid capacity |
| T7 | Deep-Tech Robotics Campus | Machinery/automotive LQ, university R&D, CORDIS EIC funding, HRST |
| T8 | Quantum / Photonics Research Anchor | Total patent density proxy, EuroHPC proximity, national quantum programme flag |

## Structural Hypotheses (Registered — Popper Contract)
Pre-registered falsifiable conjectures. Status: Registered P9; testable on 2025 Eurostat vintage and Horizon Europe 2023-2027 award data.

**RQ1 — Archetype-type alignment:** Do A2 (Frontier Innovation) regions exhibit systematically higher composite suitability indices for T1 (AI/ML) and T2 (Biotech) than A0, A1, A3 regions after controlling for log GDP per capita and log population? Falsifier: no statistically significant gradient across archetypes (Mann-Whitney p > 0.05 across all three pairwise comparisons with A2).

**RQ2 — Corridor structural advantage:** Do NUTS2 regions in a queen-contiguous Tier-1 corridor of ≥3 members show higher median structural readiness than isolated Tier-1 regions of the same archetype and type? Falsifier: Mann-Whitney U p > 0.10 for any type where both groups have n ≥ 4.

**RQ3 — Proxy reliability for T6/T8:** Does the T6 composite index correlate ρ > 0.50 with directly observed NUTS2 nuclear industry employment (NACE C24.46) in FR, FI, CZ — the three member states where that data exists? A null result (ρ < 0.50) requires proxy revision before any T6 shortlist is published externally.

**RQ4 — Country-broadcast inflation in T4:** Does the T4 Tier-1 shortlist shrink materially and become more geographically concentrated when country-level broadcast features are replaced by NUTS2-level renewable generation capacity data (EEA NUTS2 energy datasets, expected 2026)? Pre-registered direction: smaller, more concentrated shortlist.

**RQ5 — Temporal stability:** Do top-quintile regions per type on 2022-reference data remain top-quintile when recomputed on the 2024 Eurostat release? Instability > 20% per type triggers re-weighting review.

**RQ6 — I-O matrix symmetry:** Is T3↔T7 structural co-dependence symmetric (Spearman ρ > 0.60 in both directions across 242 regions)? A ρ < 0.60 in either direction means the expert-coded bidirectional link overstates structural equivalence.

## Data Architecture
- **Bronze**: Raw files as received in `data/bronze/` — immutable
- **Silver**: Harmonised NUTS2-aligned panel in `data/silver/`
  - Upstream: EU-Innovation-Panel Gold (66 features)
  - New: electricity prices, renewable share, grid proxies, water index, etc.
- **Gold**: Silver + k=4 cluster assignments + 8 suitability type scores + uncertainty

## Primary Key
`nuts2_code` — NUTS 2021 vintage, format `^[A-Z]{2}[A-Z0-9]{2}$`

## Seed Policy
- `np.random.seed(42)` at module top level
- `random_state=42` on all sklearn estimators
- All randomness logged via structlog `pipeline_step()` context manager

## Language Contract
**Permitted** (strategic intelligence framing):
- "structurally favourable", "high-index regions", "strong candidate", "above threshold"
- "shortlisted", "composite suitability index ≥ 0.70", "ranked by SC5 index"
- "structural pre-conditions associated with", "historically co-located with"
- "high-index regions for Type T1 based on [specific features]"

**Forbidden** (analytical integrity + legal hygiene):
- "the best region" (unqualified absolute superlative)
- "optimal location" (implies exhaustive optimization not performed)
- "effect of X on Y" / "causes" (causal claims; observational data only)
- "guarantees returns/success/growth"
- "will outperform / will succeed" (deterministic forecast)
- "composite ranking" (use "composite suitability index" or "type index")
- "site selection score" (use "suitability index")

## DATA GAP Handling
Known structural gaps documented in `analysis/p1_data_catalog.json`.

### Structural / Domain DATA GAPs
- **Quantum/robotics patents at NUTS2** (IPC sub-class): DATA GAP.
  Proxy: total EPO patent density + EuroHPC site proximity + national quantum flag.
  Label: `uncertainty_class = DATA_GAP_proxy`, `uncertainty_score = high`
- **Grid headroom per NUTS2**: DATA GAP (no public NUTS2 dataset).
  Proxy: ENTSO-E country NTC capacity + electricity price (inverse).
  Label: `uncertainty_class = DATA_GAP_proxy`
- **Industrial land cost**: DATA GAP (MSCI/CBRE are licensed).
  Proxy: inverse GDP_per_capita_pps + urban/rural typology flag.
  Label: `uncertainty_class = DATA_GAP_proxy`
- **HALEU patent IPC at NUTS2**: DATA GAP.
  Proxy: existing nuclear capacity + chemical industry LQ + regulatory flag.
  Label: `uncertainty_class = DATA_GAP_proxy`
- **Port throughput at NUTS2** (null rate 92.2%): DATA GAP. Excluded entirely.
  Label: `uncertainty_class = DATA_GAP_excluded`

### Country-Level Broadcast Features (variance scope: country)
These features eliminate all within-country NUTS2 variance and must not be reported
with sub-country precision. Flag in weight registry as `variance_scope: country`.
- `doing_business_score` (World Bank DB 2020)
- `cit_rate_pct` (OECD Tax Database 2023)
- `renewable_energy_share_pct` (EEA REN-share 2022)
- `ntc_import_mw` (ENTSO-E NTC 2023)
- `gerd_gov/hes/def_pct_gdp` (Eurostat rd_e_gerdfund 2021)

### Imputation Contract
- Country-median imputation compresses within-country variance to zero for imputed regions.
  Any feature imputed this way must be tagged `uncertainty_class: imputed_country_median`
  in the data catalog and may not be cited at sub-country precision.
- `ultrafast_broadband_pct` (30.2% null): imputed from country medians.
  All imputed values are flagged in the Silver frame via `_imputed_flag` columns.
- The confidence half-width multiplier (currently 0.10) is a design parameter, not a
  statistical estimate. Its value and rationale must appear in every P4 gate check output.
  Sensitivity runs at multipliers 0.10, 0.12, 0.15 are required in P4.

### Institutional & Governance DATA GAPs (Phase P11b placeholders)
These variables are currently absent or country-level only. They represent the
unobserved institutional capacity of regional actors to mobilise structural assets.
Register as `governance_relevance: high` in the data catalog.
- `rda_capacity_index`: active ESIF co-financed programmes per NUTS2 × administrative tier.
  Source: ESIF dashboard. Current resolution: NUTS2 (partial). Target: P11b.
- `institutional_diversity_score`: universities + research institutes + applied tech
  centres per million population. Source: Eurostat educ_uoe + JRC RISIS. Target: P11b.
- `regional_fiscal_autonomy_pct`: own-revenues share of regional budget.
  Source: OECD Fiscal Decentralisation Database. Current resolution: country-level only.
  Label: DATA GAP at NUTS2. Target: P11b.
- `smart_spec_strategy_flag`: active S3 strategy (JRC S3 Platform). Approximable now.
  Target: P1b extension.
- `cluster_org_count`: nationally/EU-designated industrial clusters (European Cluster
  Observatory). Approximable now. Target: P11b.

## Phase Plan
| Phase | Description | Status |
|-------|-------------|--------|
| P0 — Scaffold | Structure, CLAUDE.md, src/utils, Dockerfile, environment | PASS (GATE_P0=PASS; upstream Gold 242×66 confirmed) |
| P1 — Data Inventory | Catalog all sources: upstream Gold + 12 new datasets; DATA GAP register. **Acceptance: each feature tagged with uncertainty_class and governance_relevance.** | PASS (89 entries; 66 upstream; 16 to download; 7 DATA GAPs=7.9%; GATE_P1=PASS) |
| P1b — Downloads | Auto-fetch + static tables for all 16 new datasets | PASS (16/16; Eurostat×7, EEA live, ESIF 75K rows, static: IAEA PRIS 38 reactors, EuroHPC 18 sites, quantum flags, nuclear policy; GATE_P1b=PASS) |
| P2 — Ingestion & Harmonisation | Load upstream Gold + new datasets; k=4 clustering; build Silver+Gold. **Acceptance: output variance decomposition table (within-country vs. between-country) for all features; flag >95% between-country features as country-level broadcasts.** | PASS (re-run 2026-09-26; Silver 242x84; Gold 242x103; 18 new features; 0.0% null; k=4 sil=0.2413 ARI-vs-k2=0.2923; A0:53 A1:65 A2:43 A3:81; GATE_P2=PASS. Two real bugs fixed here, found by P3's EDA: `parse_rail_freight()` read tran_r_rapa.csv (Eurostat's rail *passengers* dataset -- a one-letter dataset-code mixup with the correct tran_r_rago goods series) and its geo-code filter matched zero rows against the wrong dataset's country-only codes, silently producing an all-zero feature via `.fillna(0.0)`; `parse_port_throughput()`'s raw file has no `geo` column at all, so a fallback silently read the constant `dataflow` string instead, extracting "ES" for every row and collapsing 26 of 27 countries into one EU-median-imputed constant. Both fixed at the source (fetched the correct tran_r_rago.csv; fixed the geo column to `rep_mar`); see scripts/p2_ingest_harmonise.py inline comments for full detail.) |
| P3 — EDA (Extended) | EDA on extended feature set; spatial autocorrelation for type scores | PASS (2026-09-26: spatial autocorrelation for type scores was already covered by P6's Moran's I/LISA, so `scripts/p3_eda_extended.py` scopes to distributions/correlation/missingness/low-variance detection on the 18 new raw features. Genuine finding: port_throughput_ktonnes and rail_freight_ktonnes both flagged as broken -- n_unique=1 across all 242 regions -- traced to real P2 ingestion bugs (see P2 row); both are actively used in P4's suitability formulas, not just a cosmetic EDA finding. GATE_P3=PASS) |
| P4 — Suitability Score Construction | 8 type composite indices; weighted scoring; uncertainty propagation. **Add: archetype×type median index table for RQ1. Add: sensitivity runs at multiplier 0.10/0.12/0.15.** | PASS (re-run 2026-09-26 after the P2 rail/port fix below; 8 types; per-type min-max rescaled to [0,1]; threshold=0.60; T1:49 T2:46 T3:75 T4:77 T5:21 T6:38 T7:31 T8:106 shortlisted -- T3 rose from 35 to 75 as a direct, real consequence of the port_throughput/rail_freight fix, not noise; GATE_P4=PASS) |
| P5 — Shortlist & Gap Analysis | Threshold filtering (≥0.70); structural gap quantification; confidence bands. **Add: corridor_isolated_comparison table (RQ2 test). Annotate gap matrix with variance_scope per feature.** | PASS (re-run 2026-09-26; Tier1≥0.70: T1:25 T2:19 T3:33 T4:52 T5:13 T6:16 T7:8 T8:53; gap matrix 308 rows; SR_med 0.75–0.89; GATE_P5=PASS) |
| P6 — Spatial Analysis | Spatial clustering of shortlists; corridor identification; LISA per type. **Add: corridor_sr_median and isolated_sr_median_same_type columns to corridors.csv. Add: corridor_name field.** | PASS (re-run 2026-09-26; Moran I 0.56–0.89 all p<0.01; 8/8 types spatially clustered; 23 corridors, 9 cross-border; GATE_P6=PASS) |
| P7 — I-O & Feasibility Framework | HALEU-SMR-DC-DeepTech I-O matrix; threshold gates; scenarios. **Add: empirical Spearman correlation table for all type pairs (RQ6). Annotate io_matrix.csv with basis column (expert_coded / empirically_supported / empirically_unsupported).** | PASS (re-run 2026-09-26; 14 I-O links, 4 strong; T6 hard-stop gate; ScenA: T1:25 T2:16 T3:23 T4:45 T5:13 T6:16 T7:7 T8:53; GATE_P7=PASS) |
| P8 — Dashboard | Palantir-style operational intelligence dashboard; 8-type explorer; I-O viewer | PASS (5-tab Dash app; 242×102 master frame; GeoJSON choropleths; GATE_P8=PASS. Not re-run 2026-09-26 -- interactive Streamlit/Dash app, not a batch script; will pick up the corrected data on next launch since it reads the regenerated Gold/analysis artifacts directly.) |
| P9 — Report | Executive intelligence brief + methodology annex; Socratic panel review applied | PASS (re-run 2026-09-26; 3,649 words; 8 sections; 3 named boxes: Popper, Deming, Scale; T3 Tier-1 count in the report body now correctly reads 33 (14%), matching the corrected P5 output; GATE_P9=PASS) |
| P10 — Reproducibility Audit | Seed audit, artifact hashing, Docker sealed run, REPRODUCE.md | PASS (2026-09-26: `scripts/p10_audit.py` built — seed audit, vocab guard, schema check, content-hash verification all green; also fixed a real Makefile bug where `feasibility` pointed at a nonexistent `p7_io_feasibility.py` instead of the real `p7_feasibility.py`. Docker sealed run not validated this pass — REPRODUCE.md documents that as an open item. Hash baseline regenerated 2026-09-26 after P11b appended governance columns to Gold — `expected_hashes.json` now covers megacampus_gold.parquet + megacampus_governance.parquet + suitability_scores.parquet, `python scripts/p10_audit.py` returns PASS.) |
| P11b — Institutional & Governance Extension | Append governance/institutional variables: rda_capacity_index, institutional_diversity_score, regional_fiscal_autonomy_pct, smart_spec_strategy_flag, cluster_org_count. All descriptive, non-causal. | PARTIAL PASS (242/242 rows; 0/5 resolved at NUTS2; 1/5 country-level broadcast proxy (institutional_diversity_score, tertiary enrolment headcount from Eurostat educ_uoe_enrt01 -- NOT a true institution-diversity measure); 4/5 DATA GAP: rda_capacity_index [ESIF dashboard: guessed Socrata view id 404'd; a working alternative dataset was found at cohesiondata.ec.europa.eu/api/views/igzp-yde2 but is a country/CCI-level programme list with no NUTS2 field, so it cannot resolve the NUTS2-level count the variable requires], smart_spec_strategy_flag [JRC S3 Platform export URL returns the Liferay portal's HTML page, not a CSV -- no bulk-export endpoint found], cluster_org_count [European Cluster Collaboration Platform export URL 404s; no public CSV/API export located], regional_fiscal_autonomy_pct [OECD SDMX dataflow id for the Fiscal Decentralisation Database could not be confirmed -- guessed dataflow 404'd]. institutional_diversity_score is flagged `institutional_diversity_score_imputed_flag=True` and gap_report status `COUNTRY_BROADCAST` (not `OK`) because it degrades to a country-level mean broadcast across NUTS2 regions when NUTS2 coverage is <20 rows -- previously this was mislabeled as a resolved, non-imputed NUTS2 value; fixed 2026-09-26. Gold pre-existing columns confirmed byte-identical; P4 T1-T8 counts unchanged from baseline (T1:49 T2:46 T3:75 T4:77 T5:21 T6:38 T7:31 T8:106); GATE_P11b=PASS on all 5 structural criteria -- row_count, columns_present, gold_unchanged, data_gap_register, p4_rerun_unchanged -- since gaps are honestly recorded rather than fabricated. `append_to_gold` is now idempotent: a re-run drops and replaces this script's own prior governance columns instead of raising on collision; `make pipeline` now includes the `governance` target.) |
| P12 — NUTS3 Down-Scaling (Pilot) | Down-scale 3–5 Tier-1 NUTS2 pilot regions to NUTS3 using Eurostat NUTS3 + Urban Audit. Priority: T5 (grid/water) and T7 (industrial district precision). | PASS (2026-09-26, corrected after final review caught unit-mismatch bugs in the first pass: SE11 Stockholm, FI1B Helsinki-Uusimaa, DK01 Hovedstaden down-scaled. Of 15 combined T5+T7 features, 2 are honestly resolved at NUTS3 — `epo_patents_per_mio_pop` (Eurostat `pat_ep_rtot`, unit=P_MHAB, latest year; confirmed exact-match against parent Gold for SE11/FI1B's single-child regions) and `lq_nace_c26` as a real NACE-section-C location quotient (`nama_10r_3empers`, wstatus=EMP), flagged `status: PROXY` (not GENUINE) since it's a coarser measurement than NUTS2's 2-digit LQ. `artificial_land_pct` was found NOT genuinely resolvable — `reg_area3`'s only codes are land-vs-water area ratios (~93-98%), not artificial-land share — and correctly degrades to BROADCAST. Remaining 12 features broadcast from the parent NUTS2's Gold value with `_is_nuts2_broadcast` flags. T5 weight-coverage: 0% genuine (its only attempt degraded honestly); T7: 29.4%. SE11 and FI1B genuinely have exactly 1 NUTS3 child each in the current nomenclature (confirmed against 2 independent Eurostat datasets); only DK01 has multiple (4), where down-scaling differentiates sub-regions. A parent-consistency sanity check (spec's Gate criterion 3) is now computed and reported. Standalone `data/gold/p12_nuts3_pilot.parquet` + `analysis/p12_nuts3_ranking.csv`, `megacampus_gold.parquet` confirmed byte-unchanged via real hash comparison against `expected_hashes.json`; GATE_P12=PASS.) |

## Paths
All paths relative to project root (`EU-MegaCampus-Siting/`).
Use `pathlib.Path(__file__).parent.parent` to resolve root.

## Naming Conventions
- Scripts: `p{phase}_{description}.py`
- Figures: `p{phase}_{description}.png` — `figures/` at 300 DPI
- Analysis artifacts: lowercase, underscores, in `analysis/`
