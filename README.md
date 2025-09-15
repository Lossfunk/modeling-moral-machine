# Moral Machine Repository

This repository explores data-driven approaches to moral decision modeling. Below is a summary of the main notebooks:

- **preprocessing.ipynb**: CSV from Moral Machine Data, filtered to get a subset
- **data_preparation.ipynb**: Filtered CSV -> final data tensor of shape (N, 2, D) where N is number of samples and D is the dimension of the feature vector
- **experiments_1.ipynb**: non-convex optimization methods + jury experiment
- **experiments_2.ipynb**: another jury method, but with aim to map all peaks in data (uses only cma-es)
- **linear_regression.ipynb**: converted accuracy to loss, trying to fit a linear model
- **transformers_experiment.ipynb**: initial experiment to train a transformer architecture to fit this data + some brief exploration with attention

## Notes
- a subset of data can be viewed at `sample_filtered_responses_sorted.csv`
- The original data is available on the Moral Machine Repository: check the README at https://osf.io/wt6mc/?view_only=4bb49492edee4a8eb1758552a362a2cf for details on columns and what they mean.