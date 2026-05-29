"""
Visualization for facial emotion recognition benchmark results.

Generates seven figures saved to the figures/ directory:
    Fig1_Model_Accuracy_Comparison.png
    Fig2_Model_Runtime_Comparison.png
    Fig3_CV_Accuracy_Comparison.png
    Fig4_Runtime_Vs_Accuracy.png
    Fig5_Accuracy_Stability_Comparison.png
    Fig6_Runtime_Vs_Accuracy_Vs_Stability.png
    Fig7_Model_Inference_Time_Comparison.png
    average_face.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

COLORS = ["gray", "purple", "blue", "green", "orange"]
FIGURES_DIR = "figures"


def _save(filename):
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, format="png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_average_face(avg_face, img_size=(48, 48)):
    plt.imshow(avg_face.reshape(img_size), cmap="gray")
    plt.title("Average Face")
    plt.axis("off")
    _save("average_face.png")


def plot_all(results):
    """
    Generate all seven benchmark figures from a list of result dicts.

    Parameters
    ----------
    results : list of dicts from models.train_and_evaluate()
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    models         = [r["name"]           for r in results]
    accuracies     = [r["accuracy"]        for r in results]
    runtimes       = [r["runtime"]         for r in results]
    cv_means       = [r["cv_mean"]         for r in results]
    cv_stds        = [r["cv_std"]          for r in results]
    inference_times = [r["inference_time"] for r in results]

    # Fig 1 — Accuracy comparison
    plt.bar(models, accuracies, color=COLORS)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    _save("Fig1_Model_Accuracy_Comparison.png")

    # Fig 2 — Runtime comparison
    plt.bar(models, runtimes, color=COLORS)
    plt.title("Model Runtime Comparison")
    plt.ylabel("Runtime (seconds)")
    _save("Fig2_Model_Runtime_Comparison.png")

    # Fig 3 — Cross-validation accuracy with std
    plt.figure(figsize=(8, 6))
    plt.bar(models, cv_means, yerr=cv_stds, capsize=5, color=COLORS)
    plt.title("Cross-Validation Accuracy with Standard Deviation")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    _save("Fig3_CV_Accuracy_Comparison.png")

    # Fig 4 — Accuracy vs runtime (grouped bars)
    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, accuracies, width, label="Accuracy (%)", color="blue")
    ax.bar(x + width / 2, runtimes,   width, label="Runtime (seconds)", color="orange")
    ax.set_xlabel("Models")
    ax.set_title("Accuracy vs. Runtime for Each Model")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    _save("Fig4_Runtime_Vs_Accuracy.png")

    # Fig 5 — Accuracy stability (lower std = more stable)
    plt.figure(figsize=(8, 6))
    plt.barh(models, cv_stds, color=COLORS)
    plt.title("Accuracy Stability (Lower is Better)")
    plt.xlabel("Standard Deviation of Accuracy (%)")
    _save("Fig5_Accuracy_Stability_Comparison.png")

    # Fig 6 — Trade-offs scatter (runtime vs cv_mean, size = stability)
    plt.figure(figsize=(10, 6))
    sizes = [50 / s if s > 0 else 50 for s in cv_stds]
    plt.scatter(runtimes, cv_means, s=sizes, c=COLORS, alpha=0.7)
    for i, model in enumerate(models):
        plt.annotate(model, (runtimes[i], cv_means[i]), fontsize=12)
    plt.title("Trade-Offs Between Accuracy, Runtime, and Stability")
    plt.xlabel("Runtime (seconds)")
    plt.ylabel("Cross-Validation Accuracy (%)")
    plt.grid(True)
    _save("Fig6_Runtime_Vs_Accuracy_Vs_Stability.png")

    # Fig 7 — Inference time comparison
    plt.bar(models, inference_times, color=COLORS)
    plt.title("Model Inference Time Comparison")
    plt.ylabel("Inference Time (seconds)")
    _save("Fig7_Model_Inference_Time_Comparison.png")
