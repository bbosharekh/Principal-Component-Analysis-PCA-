# Principal-Component-Analysis-PCA-

A lightweight, NumPy-only implementation of Principal Component Analysis (PCA) built from the ground up to explore the underlying linear algebra and eigenvalue decomposition behind dimensionality reduction.

## 📌 Overview
Most data scientists use `sklearn.decomposition.PCA` as a black box. This project rebuilds the algorithm from scratch to demonstrate a deep understanding of:
- Covariance matrix computation
- Eigendecomposition (Eigenvectors & Eigenvalues)
- Matrix projection and data centering

## 📊 Results & Validation
To prove the math works, I tested this custom PCA against the industry-standard Scikit-learn implementation using the [Dataset Name, e.g., Iris Dataset]. 


Both models successfully reduced the feature space to 2 dimensions with identical explained variance ratios.

## 💻 Quick Start

**Dependencies:** `numpy`

```python
import numpy as np
from pca import PCA

# 1. Initialize custom PCA
pca = PCA(n_components=2)

# 2. Fit to your dataset (X) and transform
pca.fit(X)
X_reduced = pca.transform(X)

# 3. Check the explained variance
print("Explained Variance:", pca.explained_variance_ratio_)
