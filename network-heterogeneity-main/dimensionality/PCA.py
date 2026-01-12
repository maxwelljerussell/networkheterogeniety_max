import numpy as np
import matplotlib.pyplot as plt


def pca_dimensionality_analysis(
    X: np.ndarray,
    variance_thresholds: list[float] = [0.9, 0.95, 0.99],
) -> dict:
    """
    PCA-based dimensionality analysis on spectra matrix X.

    Args:
        X: shape (n_samples, n_features); each row is one spectrum
        variance_thresholds: cumulative variance levels

    Returns:
        dict with:
            - eigenvalues
            - explained_variance_ratio
            - cumulative_explained_variance
            - effective_dims (for each threshold)
            - participation_ratio
    """
    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape

    if n_samples < 2:
        raise ValueError("Need at least 2 samples for PCA.")

    # Center rows
    X_centered = X - X.mean(axis=0, keepdims=True)

    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    eigenvalues = (S**2) / (n_samples - 1)

    total_var = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_var
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    # Effective dimensions for given thresholds
    effective_dims = {}
    for thr in variance_thresholds:
        idx = np.searchsorted(cumulative_explained_variance, thr) + 1
        effective_dims[thr] = int(idx)

    # Participation ratio: (Σ λ)^2 / Σ λ²
    participation_ratio = (total_var**2) / np.sum(eigenvalues**2)

    return {
        "eigenvalues": eigenvalues,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_explained_variance": cumulative_explained_variance,
        "effective_dims": effective_dims,
        "participation_ratio": participation_ratio,
    }

def plot_scree(results):
    evr = results["explained_variance_ratio"]
    cev = results["cumulative_explained_variance"]

    plt.figure(figsize=(7,4))
    plt.plot(evr, "o-", label="Explained variance ratio")
    plt.plot(cev, "s--", label="Cumulative explained variance")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance explained")
    plt.title("PCA Scree Plot")
    plt.grid(True)
    plt.legend()
    plt.show()
