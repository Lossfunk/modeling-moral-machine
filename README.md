# Building Interpretable Models for Moral Decision-Making

## Abstract

We build a custom transformer model to study how neural networks make moral decisions on trolley-style dilemmas. The model processes structured scenarios using embeddings that encode who is affected, how many people, and which outcome they belong to. Our 2-layer architecture achieves 77% accuracy on Moral Machine data while remaining small enough for detailed analysis. We use different interpretability techniques to uncover how moral reasoning distributes across the network, demonstrating that biases localize to distinct computational stages among other findings.

Preprint: [https://arxiv.org/pdf/2602.03351](https://arxiv.org/pdf/2602.03351)

## Repository Structure
- `training/`: Code for fetching the data, preprocessing it and training it
- `interp/`: Code for interpretability analyses
- `misc/`: Miscellaneous experiments ran that did not make it to the final paper

## Setup

You can use `requirements.txt` to set up a virtual environment with the necessary dependencies. 
