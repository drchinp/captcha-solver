import os
import re
import numpy as np
from PIL import Image

# --- Configuration ---
BASE_DIR = "/Users/drchinp/Library/CloudStorage/GoogleDrive-patrick@clairvoyantlab.com/My Drive/My Education/IMDA/captcha_solver_package"
IMAGE_DIR = os.path.join(BASE_DIR, "raw_images")
LABELS_DIR = os.path.join(BASE_DIR, "output")


# --- Helper function to load and flatten pixel data ---
def image_to_vector(image_path):
    """Opens an image, converts to grayscale, and flattens into a vector."""
    with Image.open(image_path) as img:
        # Convert to grayscale and then to a NumPy array of floats
        gray_img = img.convert('L')
        return np.array(gray_img, dtype="float").flatten()


# --- 1. Load the Reference Dataset ---
print("Loading reference dataset from image files...")
reference_vectors = {}
image_files = os.listdir(IMAGE_DIR)

for file_name in image_files:
    # Get the number (e.g., '00' from 'input00.jpg')
    match = re.search(r'(\d+)', file_name)
    if not match: continue
    number_part = match.group(1)

    # Get the correct label
    label_path = os.path.join(LABELS_DIR, f"output{number_part}.txt")
    if not os.path.exists(label_path): continue
    with open(label_path, 'r') as f:
        label = f.read().strip()

    # Load the image and convert to a vector
    image_path = os.path.join(IMAGE_DIR, file_name)
    if os.path.exists(image_path):
        reference_vectors[image_path] = {
            "label": label,
            "vector": image_to_vector(image_path)
        }

print(f"Loaded {len(reference_vectors)} reference images.")

# --- 2. Select and Process a "New" Image to Test ---
# Use input100.jpg as our test case
TEST_IMAGE_PATH = os.path.join(IMAGE_DIR, "input100.jpg")
print(f"\nUsing '{os.path.basename(TEST_IMAGE_PATH)}' as the new image to predict.")
new_image_vector = image_to_vector(TEST_IMAGE_PATH)

# --- 3. Find the Most Similar Image ---
best_match_image = None
smallest_difference = float('inf')

for path, data in reference_vectors.items():
    # Don't compare the image to itself
    if path == TEST_IMAGE_PATH:
        continue

    # Calculate the Mean Squared Error (a measure of difference)
    difference = np.mean((new_image_vector - data["vector"]) ** 2)

    if difference < smallest_difference:
        smallest_difference = difference
        best_match_image = data
        best_match_image["path"] = path

# --- 4. Make the "Prediction" ---
if best_match_image:
    predicted_label = best_match_image["label"]
    print("\n--- Prediction Complete ---")
    print(f"The most similar image in the dataset is: '{os.path.basename(best_match_image['path'])}'")
    print(f"Therefore, the predicted label is: '{predicted_label}'")
else:
    print("Could not find a suitable match.")