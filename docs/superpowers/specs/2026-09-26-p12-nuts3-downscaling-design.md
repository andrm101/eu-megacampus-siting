# EU-MegaCampus-Siting P12: NUTS3 Down-Scaling Pilot — Design

## Goal
Down-scale 3 Tier-1 NUTS2 regions (SE11 Stockholm, FI1B Helsinki-Uusimaa,
DK01 Hovedstaden) to their constituent NUTS3 sub-regions, for T5
(Hyperscale Data Centre Hub) and T7 (Deep-Tech Robotics Campus) —
the two types this pilot is chartered to sharpen. Output identifies which
specific NUTS3 sub-region within each parent NUTS2 actually drives the
parent's high suitability score.

## Feature-level NUTS3 resolvability (assessed up front, not discovered by failed downloads)

T5 formula (weight, direction): `electricity_price_eur_kwh`(3,−),
`ntc_import_mw`(3,+), `renewable_energy_share_pct`(3,+),
`ultrafast_broadband_pct`(3,+), `water_exploitation_index`(2,−),
`artificial_land_pct`(2,+), `doing_business_score`(2,+), `score_infra`(1,+).

T7 formula: `lq_nace_c26`(3,+), `score_talent`(3,+), `hrst_per_1000`(2,+),
`gerd_hes_pct_gdp`(2,+), `epo_patents_per_mio_pop`(2,+),
`score_cluster`(2,+), `doing_business_score`(2,+), `rail_freight_ktonnes`(1,+).

| Feature | NUTS3 source attempted | Expected outcome |
|---|---|---|
| `artificial_land_pct` | Eurostat `reg_area3` (land use by NUTS3) | Likely genuine NUTS3 resolution |
| `epo_patents_per_mio_pop` | Eurostat `pat_ep_rtot` (patent applications by NUTS3) | Likely genuine NUTS3 resolution |
| `lq_nace_c26` | Eurostat `nama_10r_3empers` (NUTS3 employment by NACE section) | Best-available proxy: NACE section C (manufacturing) at NUTS3, since 2-digit C26 detail is not published below NUTS2 — documented as a coarser proxy, not the exact same location quotient computed at NUTS2 |
| `electricity_price_eur_kwh`, `ntc_import_mw`, `renewable_energy_share_pct`, `ultrafast_broadband_pct`, `water_exploitation_index`, `doing_business_score`, `score_infra`, `score_talent`, `hrst_per_1000`, `gerd_hes_pct_gdp`, `score_cluster`, `rail_freight_ktonnes` | none — Eurostat does not publish these below NUTS2 in free bulk form | Broadcast from parent NUTS2 value to every child NUTS3, with `<feature>_is_nuts2_broadcast=True` |

This table is fixed at design time; if a genuine-resolution attempt 404s
or returns unusable data, it degrades to broadcast with the same honest
flag — never fabricated.

## Pipeline
New script `scripts/p12_nuts3_pilot.py`, following the P1b/P2 fetch +
harmonise pattern:
1. Fetch NUTS3 boundary/membership list for the 3 pilot NUTS2 parents
   (Eurostat NUTS 2021 correspondence table).
2. Fetch `reg_area3`, `pat_ep_rtot`, `nama_10r_3empers` for those NUTS3
   codes.
3. For each of the 3 genuinely-resolvable features, compute the NUTS3
   value; for all others, broadcast the parent NUTS2's existing Gold
   value.
4. Recompute the T5/T7 weighted composite (same formula/weights as
   `p4_suitability_scores.py`) at NUTS3 resolution.
5. Sanity check: for the 3 genuinely-resolved features, confirm the
   population-weighted NUTS3 average is within a documented tolerance of
   the real parent NUTS2 figure (a soft consistency check, not a hard
   gate — real aggregation methodology differences are expected).

## Output
- `data/gold/p12_nuts3_pilot.parquet` — NUTS3-indexed, T5/T7 composite +
  component columns + broadcast flags. **Standalone artifact**, not
  merged into `megacampus_gold.parquet` — avoids touching any existing
  gate (P4–P11b all stay numerically unchanged).
- `analysis/p12_nuts3_ranking.csv` — per parent NUTS2, its NUTS3 children
  ranked by T5/T7 composite, so the "which sub-region drives it" question
  has a direct answer.

## Gate criteria (`GATE_P12`)
1. All 3 pilot NUTS2s' full NUTS3 child sets present, no duplicates.
2. Broadcast-flag coverage honestly reported (how many of the 11
   formula-weight-points, out of 20 for T5 / 20 for T7, come from
   genuinely NUTS3-resolved vs. broadcast features).
3. Sanity-check tolerance report for the 3 genuinely-resolved features
   (not a pass/fail gate — a documented consistency figure).
4. No modification to any pre-existing Gold/analysis artifact — verified
   via the same content-hash convention as `p10_audit.py`.

## Out of scope
- Merging NUTS3 results back into the NUTS2 composite scores.
- Extending beyond the 3 named pilot regions.
- Dashboard UI changes — the pilot output is referenced by file path only,
  not wired into `p8_dashboard`'s interactive views in this phase.
