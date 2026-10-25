# Repository Guidelines

## Project Structure & Module Organization

EyeQuake is a Python research prototype with a static web MVP. Core package code lives in `src/eyequake/`: `data/` fetches and cleans catalogs, `analysis/` holds seismology and hazard logic, `forecast/` contains ETAS, baselines, evaluation, and deterministic tests, and `web/` exports browser data. Numbered workflows live in `scripts/` and should stay sequential, for example `07_fetch_afad.py` then `08_etas.py`. Tests live in `tests/test_*.py`. Static assets and Leaflet UI live in `web/`; generated API/data outputs belong under `web/data/`. Regenerated raw data, processed data, and figures are ignored under `data/` and `reports/figures/`.

## Build, Test, and Development Commands

- `python3 -m venv .venv` creates the local environment.
- `./.venv/bin/python -m pip install -e ".[dev]"` installs the package plus `pytest`, `pytest-cov`, and `ruff`.
- `./.venv/bin/python -m pytest` runs the default quiet test suite from `pyproject.toml`.
- `./.venv/bin/python -m pytest --cov=eyequake --cov-report=term-missing` matches CI coverage reporting.
- `./.venv/bin/ruff check .` runs lint checks.
- `./.venv/bin/python scripts/06_export_web_data.py` regenerates static web data after model or hazard changes.

## Coding Style & Naming Conventions

Target Python 3.11+. Use 4-space indentation, type-aware function boundaries, and `snake_case` for modules, functions, and variables. Keep constants in `UPPER_SNAKE_CASE`. Ruff is configured with line length `100` and target `py311`; do not add broad ignores without reason. Keep public research claims conservative: expose uncertainty, provenance, and baseline comparisons near forecast outputs.

## Testing Guidelines

Use `pytest`. Name files `tests/test_<area>.py` and tests `test_<behavior>`. Prefer small synthetic arrays for deterministic math tests, and add static web assertions when UI wording protects scientific honesty or safety. For forecast, hazard, or data-cleaning changes, cover edge cases such as empty history, bad magnitudes, and baseline comparisons. Coverage has no configured threshold, but CI prints missing lines.

## Commit & Pull Request Guidelines

History uses concise prefixes such as `feat:`, `fix:`, `docs:`, `test+build:`, and `rigor:`. Keep subjects under one line and describe the scientific or product effect. PRs should include purpose, changed tracks or scripts, commands run, generated files, and screenshots for `web/` changes. Link issues when available and call out data-source changes, model-claim changes, or regenerated artifacts.

## Security & Configuration Tips

Do not commit secrets, `.venv/`, raw catalogs, processed data, or generated figures. Prefer AFAD as the canonical Turkey catalog unless a comparison explicitly needs USGS. Keep frontend artifacts static and avoid implying deterministic earthquake prediction.
