# Shortcut to run.py (which also works on Windows and Colab):  make <step> PROFILE=t4 SET="key=value"
# The list of steps:  python run.py list
PY      ?= python
PROFILE ?=
SET     ?=

install:
	pip install -r requirements.txt && pip install -e .

%:
	$(PY) run.py $@ $(if $(PROFILE),--profile $(PROFILE)) $(if $(SET),--set $(SET))
