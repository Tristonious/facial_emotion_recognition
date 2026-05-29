"""
Data loading and preprocessing for facial emotion recognition.

Loads FER2013 images from disk, converts to grayscale, flattens to
pixel vectors, and normalizes to [0, 1]. Returns numpy arrays ready
for PCA and classifier training.

Dataset
-------
FER2013: 48x48 grayscale facial expression images across 7 emotion
classes. Available at: https://www.kaggle.com/datasets/msambare/fer2013
"""

import os
import numpy as np
from PIL import Image

IMG_SIZE = (48, 48)


def load_images_from_folder(base_path):
    """
    Walk a directory of class-labeled subdirectories, load all .jpg
    images as flattened grayscale pixel arrays, and return a numeric
    label mapping.

    Parameters
    ----------
    base_path : str
        Path to a split directory (train/ or test/) where each
        subdirectory name is an emotion class label.

    Returns
    -------
    images : np.ndarray, shape (n_samples, 2304)
    labels : np.ndarray, shape (n_samples,)
    label_map : dict {class_name: int}
    """
    images, labels = [], []
    label_map = {}
    label_counter = 0
    total_files = sum(len(files) for _, _, files in os.walk(base_path))
    processed = 0

    for root, _, files in os.walk(base_path):
        for file in files:
            if not file.endswith(".jpg"):
                continue
            label = os.path.basename(os.path.dirname(os.path.join(root, file)))
            if label not in label_map:
                label_map[label] = label_counter
                label_counter += 1

            img = Image.open(os.path.join(root, file)).convert("L").resize(IMG_SIZE)
            images.append(np.array(img).flatten())
            labels.append(label_map[label])

            processed += 1
            if processed % 1000 == 0:
                print(f"  Loaded {processed}/{total_files} images")

    return np.array(images), np.array(labels), label_map


def load_dataset(dataset_path):
    """
    Load and normalize train and test splits.

    Parameters
    ----------
    dataset_path : str
        Root directory containing train/ and test/ subdirectories.

    Returns
    -------
    X_train, y_train, X_test, y_test, label_map
    """
    train_path = os.path.join(dataset_path, "train")
    test_path  = os.path.join(dataset_path, "test")

    print("Loading training data...")
    X_train, y_train, label_map = load_images_from_folder(train_path)
    print(f"  {X_train.shape[0]} training images loaded.")

    print("Loading test data...")
    X_test, y_test, _ = load_images_from_folder(test_path)
    print(f"  {X_test.shape[0]} test images loaded.")

    # Normalize pixel values to [0, 1]
    X_train = X_train / 255.0
    X_test  = X_test  / 255.0

    return X_train, y_train, X_test, y_test, label_map
