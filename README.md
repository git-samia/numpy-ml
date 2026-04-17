# NumpyML

A from-scratch implementation of core machine learning algorithms using only NumPy. No scikit-learn, PyTorch, or TensorFlow for model training — just linear algebra and gradient math.

## What's Implemented

**Models**
- Linear Regression — ordinary least squares
- Ridge Regression — L2 regularization for weight shrinkage
- Lasso Regression — L1 regularization for automatic feature selection
- Softmax Classifier — multinomial logistic regression for multi-class problems

**Optimizers**
- SGD — mini-batch stochastic gradient descent
- Adam — adaptive learning rates with bias-corrected momentum
- Proximal GD — proximal gradient descent with soft-thresholding for L1 problems

**Feature Representations**
- Polynomial Features — higher-degree polynomial expansion with optional cross terms
- RBF Features — radial basis function mapping with random center selection

## Demo Notebook

`demo.ipynb` walks through three experiments:

1. **Regression** — Compares OLS, Ridge, and Lasso on the Diabetes dataset. Visualizes learning curves and shows how L1 regularization drives coefficients to zero for automatic feature selection.
2. **Classification** — Trains softmax classifiers on handwritten digits with no regularization, L2, and L1 penalties. Compares convergence and accuracy.
3. **Feature Engineering** — Fits y = sin(x) using linear, polynomial, and RBF features to show how non-linear feature mappings extend linear models.

## Getting Started

```bash
git clone https://github.com/git-samia/ml-from-scratch.git
cd ml-from-scratch
pip install -r requirements.txt
jupyter notebook demo.ipynb
```

## Project Structure

```
ml-from-scratch/
├── models.py         # Linear, Ridge, Lasso, Softmax Classifier
├── optimizers.py     # SGD, Adam, Proximal GD
├── features.py       # Polynomial and RBF feature transforms
├── trainer.py        # Training loop and evaluation metrics
├── demo.ipynb        # Interactive demo notebook
├── requirements.txt
└── README.md
```

## Tech Stack

- **Python 3** + **NumPy** — all algorithm implementations
- **Matplotlib** — visualizations
- **scikit-learn** — dataset loading and preprocessing only
