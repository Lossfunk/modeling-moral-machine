# Modeling Moral Machine

Explore the MIT Moral Machine dataset to study preferences in trolley-problem-style moral dilemmas. This project prepares compact feature representations and runs baseline models to probe which factors drive choices.

**Core idea:** Convert raw responses into a tensor of paired scenarios `(N, 2, D)` and learn/score feature weights that align with observed human judgments.

---

## Quickstart

1. **Setup environment**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -U pip jupyter pandas numpy scikit-learn torch matplotlib
   ```
2. **Run notebooks**
   - Launch: `jupyter lab` or `jupyter notebook`
   - Execute headlessly:  
     `jupyter nbconvert --to notebook --execute file.ipynb --inplace`
   - Clear outputs before commit:  
     `jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace file.ipynb`

---

## Repository Structure

- **Notebooks (root):**  
  `preprocessing.ipynb`, `data_preparation.ipynb`, `experiments_1.ipynb`, `experiments_2.ipynb`, `linear_regression.ipynb`
- **Data artifacts:**  
  Large files (`*.csv`, `*.pt`, `*.tar.gz`) in project root (ignored by Git)
- **Experiment logs:**  
  `wandb/` (optional)

---

## Data Pipeline

1. **Preprocess raw CSVs → filtered CSV**
   - Open `preprocessing.ipynb` and run all cells.
2. **Prepare tensors → `(N, 2, D)` dataset**
   - Run `data_preparation.ipynb` to build `data_tensor.pt` (or variants).
3. **Train/analyze models**
   - Use `experiments_1.ipynb` and `experiments_2.ipynb` for baselines and ablations.

**Note:**  
Large datasets are not tracked by Git. Keep them in the repo root or document external storage.  
If you lack the raw dataset, follow instructions in `preprocessing.ipynb` to reproduce filtered files.

---

## Reproducibility & Tracking

- Set random seeds in notebooks (NumPy/PyTorch) for reproducibility.
- Optional: log runs to Weights & Biases (`wandb`).  
  Set `WANDB_API_KEY` in your environment.

---

## Testing & Development

- Prefer pushing reusable logic into small modules (e.g., future `src/`) and test with `pytest`.
- Notebook cells should be small and deterministic.
- See [`AGENTS.md`](./AGENTS.md) for contributor guidelines (structure, style, testing, PR checklist).

---

## Contributing

- Adopt Conventional Commits (e.g., `feat:`, `fix:`, `docs:`) for clarity.
- PRs should include: clear description, linked issue (if any), data provenance/size notes, and screenshots of key results.
- Strip notebook outputs before committing.
- Avoid adding new large binaries unless essential and documented.

---
