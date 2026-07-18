.PHONY: check test mypy ruff integration install-dev

check: test mypy ruff
	@echo "✓ all checks passed"

test:
	pytest tests/ -x

mypy:
	mypy src/ --strict

ruff:
	ruff check src/

integration:
	python scripts/integration_race.py

install-dev:
	pip install -e ".[dev]"
