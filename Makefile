PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: setup run smoke

setup:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	streamlit run app.py

smoke:
	$(PYTHON) smoke_test.py
