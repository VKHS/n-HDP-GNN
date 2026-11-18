# Common tasks.  Examples:
#   make test
#   make smoke                             # seconds, no PyTorch: stages A and B, gate, partition, depth study
#   make smoke-full                        # minutes, needs PyTorch: every model, sweep runner, DDP, tables
#   make demo-quick                        # about a minute: the demo on a small graph, in runs/demo-quick/
#   make artifacts figures                 # measured stage A/B artifacts and their figures (no PyTorch)
#   make sweep-table1_ablation tables-table1_ablation
PY ?= python
DATASETS ?= cora citeseer pubmed photo computers

.PHONY: test smoke smoke-full demo demo-quick diagrams check-diagrams artifacts figures

test:
	$(PY) -m pytest -q

smoke:
	$(PY) scripts/smoke.py --level numpy

smoke-full:
	$(PY) scripts/smoke.py --level full

demo:
	$(PY) examples/demo.py

demo-quick:
	$(PY) examples/demo.py --quick --out runs/demo-quick/output --work runs/demo-quick/work

diagrams:
	$(PY) scripts/make_diagrams.py

check-diagrams:
	$(PY) scripts/make_diagrams.py --check

artifacts:
	$(PY) -m nhdp_gnn.artifacts --datasets $(DATASETS) --out artifacts

figures:
	$(PY) scripts/make_figures.py --artifacts artifacts

sweep-%:
	$(PY) -m nhdp_gnn.sweep configs/$*.toml

tables-%:
	$(PY) scripts/make_tables.py configs/$*.toml --out results/$*
