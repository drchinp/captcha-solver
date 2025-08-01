# train_model.py
import os
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# --- Configuration ---
BASE_DIR = "/Users/drchinp/Library/CloudStorage/GoogleDrive-patrick@clairvoyantlab.com/My Drive/My Education/IMDA/captcha_solver_package"
TRAIN_DIR = os.path.join(BASE_DIR, "segmented_images")
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILENAME = "captcha_character_model.h5"
LABEL_MAP_FILENAME = "class_indices.pkl"
IMG_WIDTH, IMG_HEIGHT = 28, 28
BATCH_SIZE = 32
EPOCHS = 20 # Increased epochs for better training on more data

# --- Setup ---
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# Data Generator with Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2, # Use 20% of data for validation
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
)

# --- Data Loading ---
print("Loading training and validation data...")
train_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# --- Model Building ---
NUM_CLASSES = len(train_gen.class_indices)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 1), padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# --- Training ---
print("Starting model training...")
model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen)
print("✅ Model training complete.")

# --- Saving ---
print("Saving model and label mapping...")
with open(os.path.join(MODEL_OUTPUT_DIR, LABEL_MAP_FILENAME), 'wb') as f:
    pickle.dump(train_gen.class_indices, f)

model.save(os.path.join(MODEL_OUTPUT_DIR, MODEL_FILENAME))
print(f"✅ Model and labels saved in '{MODEL_OUTPUT_DIR}'")
