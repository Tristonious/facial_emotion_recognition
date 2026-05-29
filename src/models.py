"""
Classifier training and evaluation for facial emotion recognition.

Trains four classifiers on PCA-reduced FER2013 features:
    - Eigenfaces 1-NN  (KNeighborsClassifier, k=1)
    - Shallow Neural Network (MLPClassifier, 1 hidden layer)
    - K-Nearest Neighbors (KNeighborsClassifier, k=3)
    - Random Forest (RandomForestClassifier, 100 trees)

Each classifier is evaluated on test accuracy, training runtime,
5-fold cross-validation accuracy (+/- std), and inference time.
A random baseline (1/7 chance for 7 classes) is included for reference.
"""

import time
import numpy as np
from sklearn.neural_network  import MLPClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import accuracy_score
from sklearn.model_selection import cross_val_score


def _evaluate(name, clf, X_train, y_train, X_test, y_test):
    """
    Fit a classifier, measure accuracy, runtime, CV, and inference time.

    Returns
    -------
    dict with keys: name, accuracy, runtime, cv_mean, cv_std, inference_time
    """
    # Training
    t0 = time.time()
    clf.fit(X_train, y_train)
    runtime = time.time() - t0

    # Test accuracy
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions) * 100

    # 5-fold cross-validation on training data
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    cv_mean = np.mean(cv_scores) * 100
    cv_std  = np.std(cv_scores)  * 100

    # Inference time (single pass over test set)
    t0 = time.time()
    clf.predict(X_test)
    inference_time = time.time() - t0

    print(f"{name}: acc={accuracy:.2f}%  cv={cv_mean:.2f}%±{cv_std:.2f}%  "
          f"train={runtime:.2f}s  infer={inference_time:.4f}s")

    return {
        "name":           name,
        "accuracy":       accuracy,
        "runtime":        runtime,
        "cv_mean":        cv_mean,
        "cv_std":         cv_std,
        "inference_time": inference_time,
    }


def train_and_evaluate(X_train, y_train, X_test, y_test, n_classes=7):
    """
    Train all classifiers and return a list of result dicts.

    Parameters
    ----------
    X_train, y_train : PCA-reduced training features and labels
    X_test, y_test   : PCA-reduced test features and labels
    n_classes        : int, number of emotion classes (default 7)

    Returns
    -------
    list of result dicts (one per model including baseline)
    """
    baseline_acc = 100.0 / n_classes

    results = [
        {
            "name":           "Random",
            "accuracy":       baseline_acc,
            "runtime":        0.01,
            "cv_mean":        baseline_acc,
            "cv_std":         0.0,
            "inference_time": 0.01,
        }
    ]

    classifiers = [
        ("Eigenfaces",    KNeighborsClassifier(n_neighbors=1)),
        ("SNN",           MLPClassifier(hidden_layer_sizes=(100,), activation="relu",
                                        solver="adam", max_iter=5000, random_state=42)),
        ("KNN",           KNeighborsClassifier(n_neighbors=3)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]

    for name, clf in classifiers:
        results.append(_evaluate(name, clf, X_train, y_train, X_test, y_test))

    return results
