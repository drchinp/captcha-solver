# segment_from_images.py

import os
import cv2
import re
import string
from tqdm import tqdm  # pip install tqdm

RAW_IMAGES_FOLDER = "raw_images"
LABELS_FOLDER = "output"
SEGMENTED_OUTPUT_FOLDER = "segmented_images"

# Allowable characters for folder names (avoid ?, *, etc.)
SAFE_CHARS = string.ascii_letters + string.digits

# Create the main output folder if it doesn't exist
os.makedirs(SEGMENTED_OUTPUT_FOLDER, exist_ok=True)

# Get a list of all image files in the raw_images folder
image_files = os.listdir(RAW_IMAGES_FOLDER)
print(f"Found {len(image_files)} images to process...")

# Process each image with a progress bar
for file_name in tqdm(image_files, desc="Segmenting images"):
    match = re.search(r'(\d+)', file_name)
    if not match:
        print(f"  [Warning] Could not find a number in '{file_name}'. Skipping.")
        continue

    number_part = match.group(1)
    label_file_name = f"output{number_part}.txt"
    label_path = os.path.join(LABELS_FOLDER, label_file_name)

    if not os.path.exists(label_path):
        print(f"  [Warning] Label file '{label_path}' not found for image '{file_name}'. Skipping.")
        continue

    with open(label_path, 'r') as f:
        label = f.read().strip()

    image_path = os.path.join(RAW_IMAGES_FOLDER, file_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"  [Warning] Could not read image '{file_name}'. Skipping.")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use Otsu's thresholding for better binarization
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find external contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = [b for b in bounding_boxes if b[2] > 2 and b[3] > 2]

    # Sort top-to-bottom, then left-to-right
    bounding_boxes.sort(key=lambda x: (x[1], x[0]))

    print(f"Processing '{file_name}' with label '{label}'. Found {len(bounding_boxes)} characters.")

    if len(bounding_boxes) != len(label):
        print(f"  [Warning] Mismatch! Found {len(bounding_boxes)} contours but label is '{label}'. Skipping.")
        continue

    for char_label, box in zip(label, bounding_boxes):
        x, y, w, h = box

        # Sanitize character label to make safe folder names
        clean_label = ''.join(c for c in char_label if c in SAFE_CHARS)
        if not clean_label:
            print(f"  [Warning] Invalid character label '{char_label}' skipped.")
            continue

        char_folder = os.path.join(SEGMENTED_OUTPUT_FOLDER, clean_label)
        os.makedirs(char_folder, exist_ok=True)

        char_image = thresh[y:y + h, x:x + w]

        # Optional: pad and resize to 28x28 (like MNIST)
        char_image = cv2.copyMakeBorder(char_image, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
        char_image = cv2.resize(char_image, (28, 28))

        count = len(os.listdir(char_folder)) + 1
        save_path = os.path.join(char_folder, f"{count}.png")
        cv2.imwrite(save_path, char_image)

print("\n✅ All images processed and characters segmented.")


