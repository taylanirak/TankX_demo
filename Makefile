# tankx-research developer convenience targets.
# On Windows, GNU Make is most easily run via `make` from Git for Windows
# or WSL. All targets also work via the equivalent commands documented in README.

.PHONY: install install-dev fmt lint type test test-fast cov data app clean

install:
	pip install -e .

install-dev:
	pip install -e .[dev]

fmt:
	ruff format src tests scripts

lint:
	ruff check src tests scripts

type:
	mypy --strict src/tankx

test:
	pytest -q

test-fast:
	pytest -q -m "not slow"

cov:
	pytest --cov=src/tankx --cov-report=term-missing --cov-report=html

data:
	python scripts/fetch_data.py --symbol BTC/USDT --timeframe 5m --days 365

app:
	streamlit run app.py

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov .hypothesis
	find . -type d -name __pycache__ -exec rm -rf {} +
