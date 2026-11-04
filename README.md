# ML From Scratch

Classic machine learning algorithms implemented **from scratch with NumPy** — no ML libraries for the models themselves. scikit-learn is used only for datasets, train/test split and evaluation metrics.

The goal: understand how each algorithm works internally (math, loss function, optimization), not just call `.fit()`.

## Algorithms

| # | Algorithm | Task | Notebook | Status |
|---|-----------|------|----------|--------|
| 1 | Logistic Regression | Binary classification | [logistic_regression.ipynb](logistic_regression.ipynb) | ✅ Done |
| 2 | Decision Tree (ID3) | Classification | — | 🔄 In progress |

## 1. Logistic Regression

- **Dataset:** synthetic data from `make_classification` (500 samples, 20 features, 2 classes)
- **Implemented:** sigmoid function, binary cross-entropy loss, batch gradient descent, bias term via a column of ones
- **Hyperparameters:** learning rate = 0.1, iterations = 1000
- **Evaluation:** precision, recall, F1-score, confusion matrix

**Results:**

| Metric | Value |
|--------|-------|
| Accuracy | ... |
| F1-score | ... |

## 2. Decision Tree (ID3) — coming soon

- **Dataset:** PlayTennis (14 samples)
- **Implemented:** entropy, information gain, recursive tree building

## Tech Stack

Python · NumPy · Matplotlib · scikit-learn (metrics only) · Jupyter

## How to Run

```bash
git clone git@github.com:gyelmuratov/ml-from-scratch.git
cd ml-from-scratch
python3 -m venv venv
source venv/bin/activate
pip install numpy matplotlib scikit-learn jupyter
```

Then open any
