# CAPTCHA Solver Package

This project explores two different methods for solving text-based CAPTCHA images. The appropriate method depends on the size of the available dataset.

1.  **Instance-Based Prediction**: A direct comparison method suitable for very small datasets where training a model is not feasible.
2.  **Convolutional Neural Network (CNN)**: A deep learning model that provides higher accuracy and better generalization, recommended for larger datasets.

---

## Approach 1: Instance-Based Prediction (For Small Datasets)

This method works by directly comparing a new CAPTCHA's characters to a reference library built from known examples. It does not require a "training" phase.

### How It Works

The `predict_by_instance.py` script executes the entire logic:
1.  **Builds a Reference Library**: It segments all known CAPTCHAs from `raw_images` into individual characters and stores them in memory.
2.  **Compares and Predicts**: It segments a new test image and compares each character against the entire reference library to find the closest visual match.

### How to Run
1.  Ensure your `raw_images` and `output` folders are populated.
2.  Run the script from your terminal:
    ```bash
    python predict_by_instance.py
    ```

### Limitations
This method is computationally slow and its accuracy is limited, as it cannot generalize to new character distortions not present in the reference set.

---

## Approach 2: Convolutional Neural Network (CNN) (Recommended for Large Datasets)

This is the standard and more powerful approach for image recognition tasks. A CNN automatically learns the features of characters from a large set of examples, allowing it to predict new, unseen CAPTCHAs with much higher accuracy.

**Note:** This method was not effective with the initial set of 24 images, as a neural network requires a substantial amount of data to learn from. The 0% accuracy result highlighted that a larger dataset is necessary for this approach to succeed.

### How It Works

This workflow involves three separate scripts:

1.  **`segment_from_images.py`**: This script preprocesses the images from `raw_images` and segments them into individual characters, creating an organized training dataset in the `segmented_images` folder.
2.  **`train_model.py`**: This script uses the dataset in `segmented_images` to train the CNN. It includes **data augmentation** to improve performance. The trained model is saved in the `models` folder.
3.  **`evaluate_model.py`**: This script tests the trained model against the raw CAPTCHAs to measure its final accuracy.

### Instructions for Use (with a large dataset)

If you have a larger dataset (e.g., 100+ images), follow these steps to train and use the CNN:

1.  **Populate Folders**: Place your image files in `raw_images` and their corresponding labels in `output`.
2.  **Clean Up**: To ensure a fresh start, delete the `segmented_images` folder and any files inside the `models` folder.
3.  **Segment the Images**:
    ```bash
    python segment_from_images.py
    ```
4.  **Train the Model**:
    ```bash
    python train_model.py
    ```
5.  **Evaluate the Model**:
    ```bash
    python evaluate_model.py
    ```