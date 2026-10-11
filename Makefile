.PHONY: help install test lint fmt data fit export calibrate pipeline api up down clean

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | expand -t22

install:            ## install Python deps (model + API)
	uv sync --extra fast

test:               ## run every test except the slow real-fit one
	uv run pytest -m "not slow"

test-all:           ## run everything, including the end-to-end BYOD fit
	uv run pytest

lint:               ## ruff
	uv run ruff check model/ api/ contract/

fmt:                ## ruff --fix
	uv run ruff check --fix model/ api/ contract/

data:               ## regenerate the synthetic demo company + the 3 launches
	uv run python -m ripple_model.synth

fit:                ## fit the demo model (~11 min: full fit + holdout refit)
	uv run python model/scripts/run_fits.py

export:             ## write posterior.json / trust.json / launches.json to web/public/model/
	uv run python model/scripts/export_demo.py

calibrate:          ## run the calibration study -> web/public/model/calibration.json
	uv run python model/scripts/run_calibration.py

fixtures:           ## regenerate the contract fixtures
	uv run python contract/make_fixture.py
	uv run python contract/make_parity_fixture.py
	uv run python contract/make_update_parity_fixture.py
	uv run python model/scripts/make_demo_scan.py

check-fixtures:     ## fail if a committed fixture no longer matches its generator
	uv run python scripts/check_fixtures.py

pipeline: data fit export calibrate  ## everything, from scratch

api:                ## run the API locally with reload
	uv run uvicorn api.app.main:app --reload --port 8000

up:                 ## run the API in Docker
	docker compose up --build -d
	@echo "API docs: http://localhost:8000/docs"

down:
	docker compose down

clean:
	rm -rf .pytest_cache .ruff_cache model/data/*.npz
