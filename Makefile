.PHONY: setup test pipeline scaffold inventory ingest eda scores shortlist spatial feasibility dashboard report audit reproduce docker-build docker-run

PROJECT_ROOT := $(shell pwd)
PYTHON := python
STREAMLIT := python -m streamlit

# ─── Environment ───────────────────────────────────────────────
setup:
	conda env create -f environment.yml --name eu-megacampus || conda env update -f environment.yml --name eu-megacampus
	pre-commit install || true

# ─── Quality ───────────────────────────────────────────────────
test:
	pytest tests/ -q --tb=short

seed-check:
	$(PYTHON) src/utils/seed_check.py scripts/

vocab-check:
	@for f in analysis/*.json analysis/*.csv reports/*.md; do \
		$(PYTHON) src/utils/vocab_guard.py "$$f" 2>&1 && echo "PASS: $$f"; \
	done

# ─── Pipeline phases ───────────────────────────────────────────
scaffold:
	$(PYTHON) scripts/p0_scaffold.py

inventory:
	$(PYTHON) scripts/p1_data_inventory.py

ingest:
	$(PYTHON) scripts/p2_ingest_harmonise.py

eda:
	$(PYTHON) scripts/p3_eda_extended.py

scores:
	$(PYTHON) scripts/p4_suitability_scores.py

shortlist:
	$(PYTHON) scripts/p5_shortlist_gap.py

spatial:
	$(PYTHON) scripts/p6_spatial_analysis.py

feasibility:
	$(PYTHON) scripts/p7_feasibility.py

dashboard:
	$(STREAMLIT) run scripts/p8_dashboard.py

report:
	$(PYTHON) scripts/p9_report.py

audit:
	$(PYTHON) scripts/p10_audit.py

governance:
	$(PYTHON) scripts/p11b_governance.py

# ─── Full pipeline ─────────────────────────────────────────────
pipeline: scaffold inventory ingest eda scores shortlist spatial feasibility report governance

# ─── Reproducibility ───────────────────────────────────────────
reproduce:
	$(PYTHON) scripts/p10_audit.py --verify-hashes

# ─── Docker ────────────────────────────────────────────────────
docker-build:
	docker build -t eu-megacampus-siting:v1 .

docker-run:
	docker run --rm \
		-v $(PROJECT_ROOT)/data:/project/data \
		-v $(PROJECT_ROOT)/analysis:/project/analysis \
		-v $(PROJECT_ROOT)/figures:/project/figures \
		-v $(PROJECT_ROOT)/reports:/project/reports \
		eu-megacampus-siting:v1

docker-pipeline:
	docker run --rm \
		-v $(PROJECT_ROOT)/data:/project/data \
		-v $(PROJECT_ROOT)/analysis:/project/analysis \
		-v $(PROJECT_ROOT)/figures:/project/figures \
		-v $(PROJECT_ROOT)/reports:/project/reports \
		eu-megacampus-siting:v1 python scripts/p0_scaffold.py
