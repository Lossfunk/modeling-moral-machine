# Modeling Moral Machine

Using the MIT Moral Machine dataset - what insights can we get? The trolley problem is a classic problem to understand morality, and the dataset is a huge crowd-sourced effort.

## Files
- preprocessing.ipynb: Downloaded csv -> filtered smaller csv so that it fits in memory 
- data_preparation.ipynb: Filtered csv -> Final tensor with shape (N, 2, D) where D is dim of feature vector
- experiments_1.ipynb: fitting a D size vector on this data using some algorithms.