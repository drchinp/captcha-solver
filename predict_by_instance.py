import os
import re
import numpy as np
import cv2
from PIL import Image

# --- Configuration ---
BASE_DIR = "/Users/drchinp/Library/CloudStorage/GoogleDrive-patrick@clairvoyantlab.com/My Drive/My Education/IMDA/captcha_solver_package"
IMAGE_DIR = os.path.join(BASE_DIR, "raw_images")
LABELS_DIR = os.path.join(BASE_DIR, "output")


# --- Helper Function to create an image from pixel data ---
# This is a placeholder since we are working from JPGs now.
# If you had the .txt files, we would parse them here.
def get_image_from_source(image_path):
    """Reads a JPG and returns a NumPy array."""
    with Image.open(image_path) as img:
        return np.array(img)


# --- PHASE 1: Build the Reference "Model" ---
print("Building reference model from known CAPTCHAs...")
reference_model = []  # This will be our list of (label, vector)

# Get all image files except our test case
all_image_files = os.listdir(IMAGE_DIR)
test_image_filename = "input100.jpg"
reference_image_files = [f for f in all_image_files if f != test_image_filename and f.endswith('.jpg')]

for file_name in reference_image_files:
    # 1. Get the image's true label
    match = re.search(r'(\d+)', file_name)
    if not match: continue
    number_part = match.group(1)
    label_path = os.path.join(LABELS_DIR, f"output{number_part}.txt")
    if not os.path.exists(label_path): continue
    with open(label_path, 'r') as f:
        correct_label = f.read().strip()

    # 2. Load and segment the image
    image_path = os.path.join(IMAGE_DIR, file_name)
    image_data = get_image_from_source(image_path)
    gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = [b for b in bounding_boxes if b[2] > 2 and b[3] > 2]
    bounding_boxes.sort(key=lambda x: x[0])

    if len(bounding_boxes) != len(correct_label):
        continue  # Skip if segmentation doesn't match label length

    # 3. Add each character to our reference model
    for char_label, box in zip(correct_label, bounding_boxes):
        x, y, w, h = box
        char_image = thresh[y:y + h, x:x + w]
        # Resize to a standard size for fair comparison
        char_image_resized = cv2.resize(char_image, (28, 28))
        # Flatten the 2D image into a 1D vector and add to our model
        reference_model.append({"label": char_label, "vector": char_image_resized.flatten()})

print(f"✅ Reference model built with {len(reference_model)} character examples.")

# --- PHASE 2: Predict the New CAPTCHA (input100.jpg) ---
print(f"\nProcessing new image: '{test_image_filename}'...")
final_prediction = ""

# 1. Load and segment the test image
test_image_path = os.path.join(IMAGE_DIR, test_image_filename)
test_image_data = get_image_from_source(test_image_path)
gray = cv2.cvtColor(test_image_data, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]
contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bounding_boxes = [cv2.boundingRect(c) for c in contours]
bounding_boxes = [b for b in bounding_boxes if b[2] > 2 and b[3] > 2]
bounding_boxes.sort(key=lambda x: x[0])

# 2. For each segmented character, find the best match in our model
for box in bounding_boxes:
    x, y, w, h = box
    new_char_image = thresh[y:y + h, x:x + w]
    new_char_resized = cv2.resize(new_char_image, (28, 28))
    new_char_vector = new_char_resized.flatten()

    best_match_label = '?'
    smallest_difference = float('inf')

    # Compare against every character in our reference model
    for ref in reference_model:
        # Calculate pixel-by-pixel difference (Mean Squared Error)
        difference = np.mean((new_char_vector - ref["vector"]) ** 2)
        if difference < smallest_difference:
            smallest_difference = difference
            best_match_label = ref["label"]

    final_prediction += best_match_label

# 3. Show the final combined answer
print("\n--- Prediction Complete ---")
print(f"Predicted text for '{test_image_filename}' is: '{final_prediction}'")