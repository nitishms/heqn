import json
import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from vit_keras import vit
from tqdm import tqdm

# Check GPU availability and set memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth for the GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Define the path to your dataset
BASE_PATH = "/$//"

def load_annotations(batch_num):
    json_path = os.path.join(BASE_PATH, f'batch_{batch_num}', 'JSON', f'kaggle_data_{batch_num}.json')
    with open(json_path, 'r') as file:
        data = json.load(file)
    print(f" [*] Loaded annotations 'JSON\\kaggle_data_{batch_num}.json'")
    return data

def load_images(batch_num):
    image_folder = os.path.join(BASE_PATH, f'batch_{batch_num}', 'background_images')
    images = []
    image_paths = []
    print(f"\n [---] Loading images from 'batch_{batch_num}\\background_images'")
    for filename in tqdm(os.listdir(image_folder)):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            image_paths.append(img_path)
    print(f" [*] Loaded {len(image_paths)} images.")
    return images, image_paths

def preprocess_images(images, target_size=(224, 224)):
    processed_images = []
    print("\n[---] Preprocessing images")
    for img in tqdm(images):
        img_resized = cv2.resize(img, target_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_preprocessed = img_rgb.astype(np.float32) / 255.0  # Normalize to [0, 1]
        processed_images.append(img_preprocessed)
    print("[*] Preprocessing images done.")
    return np.array(processed_images)

def prepare_data(batch_nums):
    all_images = []
    all_labels = []
    all_image_paths = []

    for batch_num in batch_nums:
        print(f"\n[---] Processing batch {batch_num}")
        annotations = load_annotations(batch_num)
        images, image_paths = load_images(batch_num)
        
        print(f" [*] Found {len(images)} images and {len(annotations)} annotations")
        
        # Create a dictionary mapping filenames to annotations for faster lookup
        annotation_dict = {item['filename']: item for item in annotations}
        
        # Print first 5 image filenames and first 3 annotation filenames
        print(f" [*] Sample image filenames: {[os.path.basename(path) for path in image_paths[:3]]}")
        print(f" [*] Sample annotation filenames: {[item['filename'] for item in annotations[:3]]}")
        
        matched_count = 0
        unmatched_count = 0
        for img, img_path in tqdm(zip(images, image_paths)):
            img_filename = os.path.basename(img_path)
            if img_filename in annotation_dict:
                latex = annotation_dict[img_filename].get('latex')
                if latex:  # Check if latex is not None or empty string
                    all_images.append(img)
                    all_labels.append(latex)
                    all_image_paths.append(img_path)
                    matched_count += 1
                else:
                    unmatched_count += 1
                    print(f" [!] Image {img_filename} found in annotations but has no latex")
            else:
                unmatched_count += 1
                if unmatched_count <= 5:  # Print only first 5 unmatched files to avoid cluttering the output
                    print(f" [!] Image {img_filename} not found in annotations")
        
        print(f"  [>] Matched {matched_count} images with annotations")
        print(f"  [>] Unmatched {unmatched_count} images")

    print(f"[*] Total processed images: {len(all_images)}")
    print(f"[*] Total labels: {len(all_labels)}")
    print(f"[*] Sample labels: {all_labels[:5]}")  # Print first 5 labels for inspection

    if not all_labels:
        raise ValueError("[!] No valid labels found in the dataset")

    processed_images = preprocess_images(all_images)
    return processed_images, all_labels, all_image_paths

def create_model(num_classes):
    vit_model = vit.vit_b16(
        image_size=224,
        activation='softmax',
        pretrained=True,
        include_top=False,
        pretrained_top=False,
        classes=num_classes
    )
    
    inputs = Input(shape=(224, 224, 3))
    x = vit_model(inputs)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

def main():
    # FIXME: Change range to (1, 11). Using only batch 1 for now
    BATCH_NUMS = range(1, 4)

    # Prepare data
    X, y, _ = prepare_data(BATCH_NUMS) # Here, `_` => `image_paths`

    print(f"[INFO] Shape of X: {X.shape}")
    print(f"[INFO] Length of y: {len(y)}")

    # Get unique labels and create a label-to-index mapping
    unique_labels = list(set(y))
    print(f"[INFO] Number of unique labels: {len(unique_labels)}")
    print(f"[INFO] Sample unique labels: {unique_labels[:5]}")  # Print first 5 unique labels

    label_to_index = {label: index for index, label in enumerate(unique_labels)}

    # Convert string labels to indices
    y_indices = [label_to_index[label] for label in y]

    print(f"[INFO] Sample y_indices: {y_indices[:5]}")  # Print first 5 indices

    # Convert to one-hot encoding
    num_classes = len(unique_labels)
    y_one_hot = to_categorical(y_indices, num_classes=num_classes)
    
    # Create and train the model
    model = create_model(num_classes)
    train_model(model, X, y_one_hot)
    
    # Save the model
    model.save('../models/aida_vit_model.keras')
    
    # Save the label mapping
    with open('../models/label_mapping.json', 'w') as f:
        json.dump(label_to_index, f)

if __name__ == '__main__':
    main()
