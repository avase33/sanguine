.PHONY: install dev test train serve docker clean

install:
	pip install -e .

dev:
	pip install -e ".[dev,server]"

test:
	pytest -q

train:
	python -m sanguine train --md report.md --json report.json

serve:
	uvicorn sanguine.service.api:app --reload

docker:
	docker build -t sanguine . && docker run -p 8000:8000 sanguine

clean:
	rm -f *.pkl report.md report.json
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
