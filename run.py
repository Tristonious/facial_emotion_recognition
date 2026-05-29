"""
run.py — facial emotion recognition benchmark entry point.

Loads FER2013 images, applies PCA via eigenface decomposition, trains
four classifiers, evaluates accuracy and runtime, and saves all figures
to the figures/ directory.

Usage
-----
    pip install numpy matplotlib pillow scikit-learn psutil
    python run.py

Set DATASET_PATH below to the root directory containing train/ and test/.

Note on AI assistance
---------------------
The original implementation for this project was developed as coursework
(BMI 8400). The code in this repository has been refactored with the
assistance of Claude (Anthropic) for clarity, structure, and readability.
The underlying algorithms, methodology, and analysis are my own work.
"""

import os

from src.data   import load_dataset
from src.pca    import EigenfacePCA
from src.models import train_and_evaluate
from src.viz    import plot_all, plot_average_face


# ── Configuration ─────────────────────────────────────────────────────────────

DATASET_PATH       = r"path/to/fer2013"  # <-- set this to your local FER2013 path
VARIANCE_THRESHOLD = 0.95                # fraction of variance retained by PCA
N_CLASSES          = 7                   # emotion classes in FER2013

# ──────────────────────────────────────────────────────────────────────────────


def main():
    # Load and normalize images
    X_train, y_train, X_test, y_test, label_map = load_dataset(DATASET_PATH)
    print(f"Classes: {label_map}\n")

    # PCA via eigenface method
    pca = EigenfacePCA(variance_threshold=VARIANCE_THRESHOLD)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)

    # Save average face
    os.makedirs("figures", exist_ok=True)
    plot_average_face(pca.avg_face)

    # Train and evaluate all classifiers
    results = train_and_evaluate(X_train_pca, y_train, X_test_pca, y_test, n_classes=N_CLASSES)

    # Generate figures
    plot_all(results)
    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    main()
