import os
import pickle
import re
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- Configuration ---
BASE_DIR = "/Users/drchinp/Library/CloudStorage/GoogleDrive-patrick@clairvoyantlab.com/My Drive/My Education/IMDA/captcha_solver_package"
MODEL_PATH = os.path.join(BASE_DIR, "models/captcha_character_model.h5")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "models/class_indices.pkl")
IMAGES_TO_TEST_DIR = os.path.join(BASE_DIR, "raw_images")
LABELS_DIR = os.path.join(BASE_DIR, "output")
IMG_WIDTH, IMG_HEIGHT = 28, 28  # Must be the same as in training

# --- 1. Load the Trained Model and Label Mapping ---
print("Loading model and label mapping...")
model = load_model(MODEL_PATH)

with open(LABEL_MAP_PATH, 'rb') as f:
    class_indices = pickle.load(f)
    # Create a reverse mapping from index to character
    labels_to_chars = {v: k for k, v in class_indices.items()}

# --- 2. Evaluation Loop ---
total_correct = 0
total_images = 0

image_files = os.listdir(IMAGES_TO_TEST_DIR)
print(f"Found {len(image_files)} images to evaluate.")

for image_file_name in image_files:
    # Get the correct label from the corresponding output file
    match = re.search(r'(\d+)', image_file_name)
    if not match:
        continue

    number_part = match.group(1)
    label_file_name = f"output{number_part}.txt"
    label_path = os.path.join(LABELS_DIR, label_file_name)

    if not os.path.exists(label_path):
        continue

    with open(label_path, 'r') as f:
        correct_label = f.read().strip()

    # --- 3. Process the Image (same as in segmentation) ---
    image_path = os.path.join(IMAGES_TO_TEST_DIR, image_file_name)
    image = cv2.imread(image_path)
    if image is None:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = [b for b in bounding_boxes if b[2] > 2 and b[3] > 2]
    bounding_boxes.sort(key=lambda x: x[0])

    predicted_label = ""

    # --- 4. Predict Each Character ---
    for box in bounding_boxes:
        x, y, w, h = box

        # Extract character and preprocess for the model
        char_image = thresh[y:y + h, x:x + w]
        char_image_resized = cv2.resize(char_image, (IMG_WIDTH, IMG_HEIGHT))
        char_image_normalized = char_image_resized / 255.0
        # Add batch and channel dimensions
        char_image_final = np.expand_dims(np.expand_dims(char_image_normalized, axis=0), axis=-1)

        # Get prediction from the model
        prediction = model.predict(char_image_final)
        predicted_index = np.argmax(prediction)
        predicted_char = labels_to_chars[predicted_index]

        predicted_label += predicted_char

    # --- 5. Compare and Report ---
    print(f"Image: {image_file_name}, Correct: '{correct_label}', Predicted: '{predicted_label}'")
    if predicted_label == correct_label:
        total_correct += 1
    total_images += 1

# --- 6. Calculate Final Accuracy ---
accuracy = (total_correct / total_images) * 100 if total_images > 0 else 0
print("\n--- Evaluation Complete ---")
print(f"Total Images: {total_images}")
print(f"Correctly Predicted: {total_correct}")
print(f"Accuracy: {accuracy:.2f}%")