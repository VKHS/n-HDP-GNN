# Common tasks.  Examples:
#   make test
#   make artifacts figures                 # measured stage A/B artifacts and their figures (no PyTorch)
#   make sweep-table1_ablation tables-table1_ablation
PY ?= python
DATASETS ?= cora citeseer pubmed photo computers

.PHONY: test artifacts figures

test:
	$(PY) -m pytest -q

artifacts:
	$(PY) -m nhdp_gnn.artifacts --datasets $(DATASETS) --out artifacts

figures:
	$(PY) scripts/make_figures.py --artifacts artifacts

sweep-%:
	$(PY) -m nhdp_gnn.sweep configs/$*.toml

tables-%:
	$(PY) scripts/make_tables.py configs/$*.toml --out results/$*
