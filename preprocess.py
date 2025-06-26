import os
import shutil
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input
from tqdm import tqdm
import matplotlib.pyplot as plt
import time  # ✅ Prevent crashes due to rapid processing
import tensorflow as tf  # ✅ Import TensorFlow
from augmented_data import augment_image  # ✅ Import augmentation function
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Disable TensorFlow OneDNN logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Define paths
DATA_DIR = "./"
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train-images")
VAL_IMG_DIR = os.path.join(DATA_DIR, "val-images")
TRAIN_CSV = os.path.join(DATA_DIR, "train_metadata.csv")
VAL_CSV = os.path.join(DATA_DIR, "val_metadata.csv")

# Target image size (EfficientNet recommended)
IMG_SIZE = (384, 384)


# ✅ Function to delete old preprocessed data
def delete_old_data(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f"🗑 Deleted old data: {directory}")


# ✅ Load metadata CSV files
def load_metadata():
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    cols_needed = [
        "isic_id",
        "benign_malignant",
        "age_approx",
        "anatom_site_general",
        "sex",
    ]
    train_df = train_df[cols_needed]
    val_df = val_df[cols_needed]
    return train_df, val_df


# ✅ Convert labels and handle missing data
def process_metadata(df):
    """Convert labels to binary and fill missing values without chained assignment issues"""
    df = df.copy()  # Ensure we are working on a copy to avoid warnings

    # Convert labels
    df["benign_malignant"] = df["benign_malignant"].map({"benign": 0, "malignant": 1})

    # Fill missing values correctly
    df["age_approx"] = df["age_approx"].fillna(df["age_approx"].median())
    df["anatom_site_general"] = df["anatom_site_general"].fillna("unknown")
    df["sex"] = df["sex"].fillna("unknown")

    return df


def load_preprocess_image(img_path):
    # Add debug print to verify the image path
    print(f"Loading image from path: {img_path}")
    img = load_img(img_path, target_size=(384, 384))  # or your preferred method
    if img is None:
        print("❌ Failed to load image.")
    return img_to_array(img) / 255.0  # or your preferred normalization


# ✅ Process dataset with augmentation & validation
def process_dataset(df, img_dir, output_dir, is_train=True):
    delete_old_data(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 Saving preprocessed images to {output_dir}...")
    image_paths, labels, skipped_images = [], [], []
    df_copy = df.copy()  # ✅ Prevents modification issues

    for i, row in tqdm(
        df_copy.iterrows(), total=len(df_copy), desc="Processing Images"
    ):
        try:
            img_id = row["isic_id"]
            img_path = os.path.join(img_dir, f"{img_id}.jpg")

            time.sleep(0.01)  # ✅ Prevent crashes due to rapid processing
            if not os.path.exists(img_path):
                print(f"⚠️ Warning: Missing image {img_id}.jpg")
                skipped_images.append(img_id)
                continue

            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if image is None or image.size == 0:
                print(f"⚠️ Warning: Could not read {img_path} (corrupted file)")
                skipped_images.append(img_id)
                continue

            if len(image.shape) == 2 or image.shape[-1] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, IMG_SIZE)
            image = np.array(image, dtype=np.float32) / 255.0

            # ✅ Debug: Ensure normalization is within range
            image = preprocess_input(image)
            image = np.clip(image, 0.0, 1.0)

            if np.std(image) == 0:
                print(f"⚠️ Skipping Blank Image (zero variance): {img_id}")
                skipped_images.append(img_id)
                continue

            npy_path = os.path.abspath(os.path.join(output_dir, f"image_{i}.npy"))
            np.save(npy_path, image)
            np.save(
                os.path.join(output_dir, f"label_{i}.npy"),
                np.array([row["benign_malignant"]], dtype=np.int32),
            )
            image_paths.append(npy_path)
            labels.append(int(row["benign_malignant"]))

        except Exception as e:
            print(f"❌ Error processing {img_id}: {e}")

    processed_df = pd.DataFrame({"image_path": image_paths, "benign_malignant": labels})
    processed_df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

    if skipped_images:
        with open(os.path.join(output_dir, "skipped_images.txt"), "w") as f:
            f.write("\n".join(skipped_images))
        print(
            f"⚠️ Skipped {len(skipped_images)} blank/corrupted images. See 'skipped_images.txt'."
        )

    print(
        f"✅ Processed {len(image_paths)} images. Metadata saved to {output_dir}/metadata.csv"
    )


# ✅ Main function
if __name__ == "__main__":
    print("🔹 Loading metadata...")
    train_df, val_df = load_metadata()
    train_df = process_metadata(train_df)
    val_df = process_metadata(val_df)

    print("🔹 Processing images...")
    process_dataset(train_df, TRAIN_IMG_DIR, "./processed/train", is_train=True)
    process_dataset(val_df, VAL_IMG_DIR, "./processed/val", is_train=False)

    print("✅ Dataset preprocessing complete!")
