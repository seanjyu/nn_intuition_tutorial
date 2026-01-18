# data.py
import numpy as np


def generate_data(dataset_type, n_samples=200, noise=0.1):
    np.random.seed(42)

    if dataset_type == "Circle":
        from sklearn.datasets import make_circles
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)

    elif dataset_type == "XOR":
        X = np.random.randn(n_samples, 2)
        y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
        X += np.random.randn(n_samples, 2) * noise

    elif dataset_type == "Spiral":
        n = n_samples // 2
        theta = np.sqrt(np.random.rand(n)) * 2 * np.pi
        r_a = 2 * theta + np.pi
        r_b = -2 * theta - np.pi
        X = np.vstack([
            np.column_stack([r_a * np.cos(theta), r_a * np.sin(theta)]),
            np.column_stack([r_b * np.cos(theta), r_b * np.sin(theta)])
        ]) + np.random.randn(n_samples, 2) * noise * 0.5
        y = np.hstack([np.zeros(n), np.ones(n)])

    else:
        # Default to clusters
        from sklearn.datasets import make_blobs
        X, y = make_blobs(n_samples=n_samples, centers=2, random_state=42)

    # # Normalize
    # X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Min-max scaling to [-1, 1]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X = 2 * (X - X_min) / (X_max - X_min + 1e-8) - 1

    # Ensure correct dtypes
    return X.astype(np.float32), y.astype(np.float32)