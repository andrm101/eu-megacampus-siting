# Reproducing the EU MegaCampus Siting Intelligence Analysis

## 1. Prerequisites

- Python 3.11, or Docker Desktop (Dockerfile already builds from `ghcr.io/osgeo/gdal:ubuntu-small-3.8.5`)
- The upstream `EU-Innovation-Panel/data/gold/region_profiles_gold.parquet` (242×66) must exist as a sibling repo checkout — this project's P2 ingestion reads it directly.
- No API keys or paid accounts required.

## 2. Full Reproduction

```bash
conda env create -f environment.yml --name eu-megacampus
conda activate eu-megacampus

make pipeline   # scaffold -> inventory -> ingest -> eda -> scores -> shortlist -> spatial -> feasibility -> report
```

Or via Docker (`docker-build` / `docker-pipeline` Makefile targets) — the Docker path builds successfully (correct `ghcr.io` GDAL registry already in place) but running the full containerized pipeline end-to-end has not been validated as part of this P10 pass; see "Known limitations" below.

## 3. Expected Outputs

| Artifact | Path | Notes |
|---|---|---|
| Gold panel | `data/gold/megacampus_gold.parquet` | 242 regions × 103 columns |
| Suitability scores | `data/gold/suitability_scores.parquet` | 242 regions × 45 columns, 8 ecosystem types |
| Shortlists & gap analysis | `analysis/p5_*.csv` | Tier-1 (≥0.70) shortlists per type |
| Spatial diagnostics | `analysis/p6_morans_i.csv`, `p6_lisa_results.csv` | Global Moran's I + LISA per type |
| Feasibility/I-O | `analysis/p7_*.csv` | I-O matrix, scenario assignments |
| Dashboard | `make dashboard` | Streamlit app, localhost default port |
| Executive brief | `reports/p9_executive_brief.md` | ~3,600-word report |

SHA-256 checksums: see `expected_hashes.json`, generated via
`python scripts/p10_audit.py --mode=generate-hashes`. Currently covers the two
Gold-layer parquets. Parquet files are hashed on canonical data content (a
deterministic CSV serialization), not raw file bytes — pyarrow embeds
non-data metadata (e.g. write timestamps) that differs between two
numerically-identical runs, confirmed directly during EU-Innovation-Panel's
own P10 audit. **Generate the reference hash in the same Python/pandas
environment you intend to verify against** — content hashing is stable across
repeated runs within one pandas version but not guaranteed identical across
different versions.

## 4. Verifying Success

```bash
make audit          # seed audit + vocab guard + schema check
make reproduce       # hash verification only (python scripts/p10_audit.py --verify-hashes)
```

A passing audit prints `"overall_status": "PASS"` and writes the same summary
to `pipeline_audit.json`.

## 5. Known limitations

- **Docker end-to-end run not validated in this P10 pass.** The image builds
  from the correct GDAL registry, but actually running `make docker-pipeline`
  through to a Gold-layer output (the way EU-Innovation-Panel's sealed
  reproduction was validated) was deferred — a good next step if full
  container-level reproducibility assurance is needed.
- **`reports/p9_executive_brief.md`'s methodology annex intentionally lists
  every forbidden phrase** (as documentation of the project's own language
  contract) after a `**Prohibited language**` marker. `scripts/p10_audit.py`
  and `scripts/p9_report.py`'s own internal gate both split the text at that
  marker before checking — do not remove that marker or the vocab guard will
  flag the annex as violating the rules it's describing.
