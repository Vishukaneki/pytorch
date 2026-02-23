import pypandoc

text = """
# Non-Linear Classification with PyTorch

This project demonstrates how to perform binary classification on a non-linearly separable dataset using PyTorch.

## 📌 Overview
We use the `make_circles` dataset from scikit-learn to generate circular data and train a neural network to classify it.

## 📂 Dataset
- Source: sklearn.datasets.make_circles
- Samples: 1000
- Features: X1, X2
- Labels: 0 and 1

## ⚙️ Technologies Used
- Python
- PyTorch
- Scikit-learn
- Pandas
- Matplotlib

## 🧠 Workflow

1. Generate dataset using make_circles
2. Visualize data distribution
3. Convert data to PyTorch tensors
4. Split into train/test sets
5. Build neural network model
6. Train model
7. Evaluate performance
8. Plot decision boundary

## 🚀 Installation

```bash
pip install torch scikit-learn pandas matplotlib
