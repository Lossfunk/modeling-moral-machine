# Repository Guidelines

## Project Structure & Modules
- Root contains notebooks: `preprocessing.ipynb`, `data_preparation.ipynb`, `experiments_*.ipynb`, `linear_regression.ipynb`.
- Large data artifacts live at repo root (e.g., `*.csv`, `*.pt`) and experiment logs in `wandb/`.
- No `src/` or `tests/` yet. If adding reusable code, create `src/` for modules and `tests/` for unit tests.

## Dev Setup, Build & Run
- Use Python 3.10+ and a virtual env:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -U pip jupyter pandas numpy scikit-learn torch matplotlib`
- Launch notebooks locally: `jupyter lab` or `jupyter notebook`.
- Execute a notebook non-interactively: `jupyter nbconvert --to notebook --execute file.ipynb --inplace`.
- Clear outputs before committing: `jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace file.ipynb`.

## Coding Style & Naming
- Python: 4-space indentation, `snake_case` for variables/functions, `UPPER_SNAKE_CASE` for constants.
- Notebook names: `topic_action.ipynb` (e.g., `experiments_2.ipynb`). Keep cells small and deterministic.
- If scripts are added, format with `black` and lint with `ruff` (configure in a follow-up PR if needed).

## Testing Guidelines
- Prefer pushing logic into functions (in `src/`) and keep notebooks for orchestration/visualization.
- Tests: `pytest` with files named `tests/test_*.py`; run `pytest -q`.
- Add minimal fixtures; aim for coverage of core data transforms and model utilities.

## Commit & PR Guidelines
- Current history uses short, lowercase messages. Please adopt Conventional Commits (e.g., `feat:`, `fix:`, `docs:`) for clarity.
- PRs should include: clear description, linked issue (if any), data provenance/size notes, and screenshots of key results.
- Strip notebook outputs; avoid committing new large binaries unless essential and documented.

## Data & Security Notes
- Large datasets are already present; do not add new large files casually. Prefer documenting download/reproduction steps.
- Use relative paths; set random seeds for reproducibility.
- Keep credentials out of notebooks. For Weights & Biases, use `WANDB_API_KEY` env var.

## Agent-Specific Instructions
- Be surgical: avoid moving large files or renaming notebooks without need.
- When adding code, prefer small, well-named modules and include a usage example in notebooks.
