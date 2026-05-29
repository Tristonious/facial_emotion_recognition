"""
PCA via the eigenface method.

Uses the "small covariance matrix" trick: instead of computing the
full n_pixels x n_pixels covariance matrix, we compute the smaller
n_samples x n_samples matrix and project eigenvectors back into pixel
space. This is significantly faster and more memory-efficient when
n_pixels >> n_samples.

Reference
---------
Turk, M. and Pentland, A. 1991. Eigenfaces for recognition.
Journal of Cognitive Neuroscience 3(1), 71-86.
"""

import numpy as np


class EigenfacePCA:
    """
    PCA via eigenface decomposition.

    Usage
    -----
    pca = EigenfacePCA(variance_threshold=0.95)
    pca.fit(X_train)
    train_coords = pca.transform(X_train)
    test_coords  = pca.transform(X_test)
    """

    def __init__(self, variance_threshold=0.95):
        """
        Parameters
        ----------
        variance_threshold : float
            Fraction of total variance to retain. Controls the number
            of eigenfaces kept. Default 0.95 retains 95% of variance.
        """
        self.variance_threshold = variance_threshold
        self.avg_face  = None
        self.eigfaces  = None
        self.n_components = None

    def fit(self, X):
        """
        Compute eigenfaces from training images.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_pixels)
        """
        # Center images by subtracting the average face
        X_T = X.T
        self.avg_face = np.mean(X_T, axis=1)
        X_centered = X_T - self.avg_face[:, np.newaxis]

        # Small covariance matrix trick: (n_samples x n_samples) instead of
        # (n_pixels x n_pixels). Eigenvectors are already in feature subspace.
        C = np.dot(X_centered, X_centered.T)
        eigvals, eigvecs = np.linalg.eig(C)

        # Sort by descending eigenvalue magnitude
        sorted_idx    = np.argsort(-np.abs(eigvals))
        eigvals       = np.abs(eigvals[sorted_idx])
        eigvecs       = eigvecs[:, sorted_idx]

        # Normalize eigenvectors
        eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0)

        # Determine how many components explain variance_threshold of variance
        explained = eigvals / np.sum(eigvals)
        cumulative = np.cumsum(explained)
        self.n_components = int(np.argmax(cumulative >= self.variance_threshold) + 1)
        self.eigfaces = eigvecs[:, :self.n_components]

        print(f"EigenfacePCA: retaining {self.n_components} components "
              f"({self.variance_threshold*100:.0f}% variance threshold).")

        return self

    def transform(self, X):
        """
        Project images into the eigenface subspace.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_pixels)

        Returns
        -------
        coords : np.ndarray, shape (n_samples, n_components)
        """
        X_centered = (X - self.avg_face).T
        return np.dot(self.eigfaces.T, X_centered).T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
