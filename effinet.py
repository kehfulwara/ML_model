# import tensorflow as tf
# import numpy as np
# import tensorflow_addons as tfa
# import pandas as pd
# import albumentations as A
# from tensorflow.keras.models import load_model
# from tensorflow.keras.optimizers import SGD
# from tensorflow_addons.optimizers import AdamW
# import cv2
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.applications.efficientnet import preprocess_input

# from tensorflow.keras.callbacks import LearningRateScheduler
# from tensorflow_addons.optimizers import AdamW

# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# from pathlib import Path
# from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
# from tensorflow.keras.metrics import Precision, Recall, AUC
# from build_model import build_model


# # ✅ Load metadata CSV
# train_csv = "./train_metadata.csv"
# val_csv = "./val_metadata.csv"

# train_df = pd.read_csv(train_csv)
# val_df = pd.read_csv(val_csv)

# print(f"✅ Train Data Loaded: {train_df.shape}")
# print(f"✅ Validation Data Loaded: {val_df.shape}")

# # ✅ Standardize column names
# train_df.columns = train_df.columns.str.strip().str.lower()
# val_df.columns = val_df.columns.str.strip().str.lower()

# # ✅ Ensure required columns exist
# required_columns = {"isic_id", "benign_malignant"}

# assert required_columns.issubset(train_df.columns), "❌ Missing columns in train_df"
# assert required_columns.issubset(val_df.columns), "❌ Missing columns in val_df"

# # ✅ Define image directories
# IMAGE_DIR = "./train-images"
# VAL_IMAGE_DIR = "./val-images"


# # ✅ Function to check if image exists
# def check_image_exists(isic_id, image_dir):
#     for ext in [".jpg", ".jpeg", ".png"]:
#         img_path = Path(image_dir) / f"{isic_id}{ext}"
#         if img_path.is_file():
#             return str(img_path)
#     return None


# train_df["image_path"] = train_df["isic_id"].apply(
#     lambda x: check_image_exists(x, IMAGE_DIR)
# )
# val_df["image_path"] = val_df["isic_id"].apply(
#     lambda x: check_image_exists(x, VAL_IMAGE_DIR)
# )

# # ✅ Drop rows where images are missing
# train_df.dropna(subset=["image_path"], inplace=True)
# val_df.dropna(subset=["image_path"], inplace=True)

# print(f"🔍 Train dataset size after cleaning: {len(train_df)}")
# print(f"🔍 Validation dataset size after cleaning: {len(val_df)}")

# # ✅ Convert labels: Benign = 0, Malignant = 1
# label_map = {"benign": 0, "malignant": 1}
# train_df["benign_malignant"] = train_df["benign_malignant"].map(label_map).astype(int)
# val_df["benign_malignant"] = val_df["benign_malignant"].map(label_map).astype(int)

# # ✅ Compute Class Weights
# unique_labels = np.unique(train_df["benign_malignant"])
# class_weights = compute_class_weight(
#     class_weight="balanced", classes=unique_labels, y=train_df["benign_malignant"]
# )
# class_weight_dict = {0: 0.7, 1: 2.2}
# # class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# print(f"🔍 Computed Class Weights: {class_weight_dict}")

# # ✅ Scalar LR for AdamW
# optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-5)


# # ✅ Cosine Decay callback (to simulate CosineDecayRestarts)
# def cosine_decay(epoch, lr_start=1e-3, lr_min=1e-6, T_max=30):
#     from math import pi, cos

#     return lr_min + 0.5 * (lr_start - lr_min) * (1 + cos(pi * (epoch % T_max) / T_max))


# lr_callback = LearningRateScheduler(cosine_decay, verbose=1)


# # ✅ Albumentations Augmentation (Optimized for Clear, Natural Lesions)
# IMG_SIZE = (384, 384)
# train_augmentation = A.Compose(
#     [
#         A.Resize(IMG_SIZE[0], IMG_SIZE[1]),  # Resize to desired size
#         A.HorizontalFlip(p=0.7),  # Horizontal flip for variation
#         A.VerticalFlip(p=0.4),
#         A.Rotate(limit=12, border_mode=cv2.BORDER_REFLECT_101, p=0.4),
#         A.Affine(
#             translate_percent=(0.0, 0.05),  # Increased translation
#             scale=(0.98, 1.05),  # Increased scale variation
#             rotate=(-10, 10),  # Increased rotation
#             shear=(-5, 5),  # Increased shear
#             p=0.4,
#         ),
#         A.RandomBrightnessContrast(brightness_limit=0.03, contrast_limit=0.05, p=0.3),
#         A.HueSaturationValue(p=0.3),
#         A.GaussNoise(p=0.02),
#         A.GridDistortion(num_steps=3, distort_limit=0.05, p=0.3),
#         A.Sharpen(
#             alpha=(0.2, 0.4), lightness=(0.8, 1.1), p=0.4
#         ),  # Sharpen lesions more
#         # Increased shift & rotation
#         A.GaussianBlur(blur_limit=(2, 2), p=0.04),
#         A.CLAHE(
#             clip_limit=2.2, tile_grid_size=(8, 8), p=0.3
#         ),  # Increased CLAHE clip limit
#     ]
# )

# val_augmentation = A.Compose(
#     [
#         A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
#     ]
# )


# def load_and_augment_image(img_path, label, is_train=True):
#     def process_image(img_path):
#         img_path = img_path.decode("utf-8")  # Ensure string format for debugging

#         # Load image using TensorFlow methods
#         img = tf.io.read_file(img_path)
#         img = tf.image.decode_jpeg(img, channels=3)
#         img = tf.image.resize(img, IMG_SIZE)  # Resize image to target size
#         img = tf.cast(
#             img, tf.uint8
#         ).numpy()  # Convert to NumPy array for Albumentations

#         # Apply augmentations based on whether it's training or validation
#         aug_pipeline = train_augmentation if is_train else val_augmentation
#         augmented = aug_pipeline(image=img)["image"]

#         return augmented.astype(np.float32) / 255.0  # Normalize image

#     # Use tf.numpy_function to allow external processing (Albumentations)
#     img = tf.numpy_function(func=process_image, inp=[img_path], Tout=tf.float32)

#     # Explicitly set shape to avoid shape mismatches in the TensorFlow data pipeline
#     img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])  # Set static shape for the image

#     # Normalize the image using EfficientNet preprocessing (ensure this is defined)
#     # img = preprocess_input(img)  # Apply preprocessing

#     # Ensure label is float32
#     label = tf.cast(label, tf.float32)

#     return img, label


# def prepare_dataset(df, batch_size=32, shuffle=False, is_train=True):
#     # Ensure the 'image_path' column exists
#     if "image_path" not in df.columns:
#         raise KeyError("'image_path' column is missing in the DataFrame")

#     # Decode byte paths into string paths if necessary (only if the paths are in bytes)
#     df["image_path"] = df["image_path"].apply(
#         lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
#     )

#     # Debugging step: Print the first few image paths to verify
#     print(df["image_path"].head())

#     # Create the dataset from the image paths and labels
#     dataset = tf.data.Dataset.from_tensor_slices(
#         (df["image_path"].values, df["benign_malignant"].values)
#     )

#     # Shuffle dataset if specified
#     if shuffle:
#         dataset = dataset.shuffle(len(df))

#     # Map the `load_and_augment_image` function to each image
#     dataset = dataset.map(
#         lambda x, y: load_and_augment_image(x, y, is_train),
#         num_parallel_calls=tf.data.AUTOTUNE,
#     )

#     # Batch and prefetch data for efficient training
#     return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# # ✅ Example usage
# train_dataset = prepare_dataset(train_df, batch_size=32, shuffle=True, is_train=True)
# val_dataset = prepare_dataset(val_df, batch_size=32, is_train=False)


# model = build_model(img_size=(384, 384, 3), dropout_rate=0.4)

# # ✅ Compile Model
# model.compile(
#     optimizer=optimizer,
#     loss=tfa.losses.SigmoidFocalCrossEntropy(),
#     metrics=["accuracy", "AUC"],
# )

# # ✅ Callbacks
# callbacks = [
#     ModelCheckpoint(
#         "./final_model.h5",
#         save_best_only=True,
#         monitor="val_loss",
#         mode="min",
#         verbose=1,
#     ),
#     ReduceLROnPlateau(
#         monitor="val_loss", factor=0.2, patience=2, verbose=1, min_lr=1e-7
#     ),
#     EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
#     lr_callback,  # ✅ Add this
# ]


# # ✅ Train Model for 30 Epochs
# history = model.fit(
#     train_dataset,
#     epochs=30,
#     initial_epoch=5,
#     validation_data=val_dataset,
#     class_weight=class_weight_dict,
#     callbacks=callbacks,  # Change verbosity to 1 or 0 to see more detailed output
# )

# # Print the training history
# print("Training History:", history.history)

# # ✅ Fine-Tuning: Unfreeze top layers of EfficientNet-B0
# print("🔁 Starting fine-tuning...")
# base_model = model.get_layer("efficientnetb0")  # Get base model

# # Unfreeze top layers (e.g., last 40 layers)
# for layer in base_model.layers[-40:]:
#     if not isinstance(layer, tf.keras.layers.BatchNormalization):
#         layer.trainable = True

# # ✅ Fine-tuning optimizer with fixed scalar learning rate
# fine_tune_optimizer = AdamW(learning_rate=1e-5, weight_decay=1e-5)

# # Recompile with fine-tuning optimizer
# model.compile(
#     optimizer=fine_tune_optimizer,
#     loss=tfa.losses.SigmoidFocalCrossEntropy(),
#     metrics=["accuracy", "AUC"],
# )

# # ✅ Continue Training for Fine-Tuning (e.g., 20 more epochs)
# fine_tune_history = model.fit(
#     train_dataset,
#     epochs=20,
#     validation_data=val_dataset,
#     callbacks=callbacks,
#     class_weight=class_weight_dict,
# )

# # Print Fine-Tuning History
# print("Fine-Tuning History:", fine_tune_history.history)

# # ✅ Combine history for complete plotting
# history.history["accuracy"] += fine_tune_history.history["accuracy"]
# history.history["val_accuracy"] += fine_tune_history.history["val_accuracy"]
# history.history["loss"] += fine_tune_history.history["loss"]
# history.history["val_loss"] += fine_tune_history.history["val_loss"]

# # ✅ Save the model after training
# model.save("./final_model.keras")  # Recommended format

# print("✅ Model saved successfully.")

# # ✅ Model Evaluation after Fine-Tuning
# eval_results = model.evaluate(val_dataset)
# print(f"Evaluation Results after Fine-Tuning: {eval_results}")

# from sklearn.metrics import (
#     classification_report,
#     confusion_matrix,
#     roc_auc_score,
#     precision_score,
#     recall_score,
#     f1_score,
# )

# # Optionally, print classification metrics
# y_true = val_df["benign_malignant"].values
# y_pred = (model.predict(val_dataset) > 0.5).astype("int32")
# precision = precision_score(y_true, y_pred)
# recall = recall_score(y_true, y_pred)
# f1 = f1_score(y_true, y_pred)

# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1-Score: {f1}")

# print(classification_report(y_true, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
# print("AUC Score:", roc_auc_score(y_true, y_pred))

# # ✅ Plot Accuracy & Loss
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(history.history["accuracy"], label="Train Accuracy")
# plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
# plt.legend()
# plt.title("Model Accuracy")

# plt.subplot(1, 2, 2)
# plt.plot(history.history["loss"], label="Train Loss")
# plt.plot(history.history["val_loss"], label="Validation Loss")
# plt.legend()
# plt.title("Model Loss")

# plt.show()

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,  # Added import
    auc,  # Import auc
)
from tensorflow.keras.models import load_model
from build_model import build_model
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import pandas as pd
import albumentations as A
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping,
)
from tensorflow_addons.optimizers import AdamW
from pathlib import Path

# ✅ Load metadata CSV
train_csv = "./train_metadata.csv"
val_csv = "./val_metadata.csv"

train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)

print(f"✅ Train Data Loaded: {train_df.shape}")
print(f"✅ Validation Data Loaded: {val_df.shape}")

# ✅ Standardize column names
train_df.columns = train_df.columns.str.strip().str.lower()
val_df.columns = val_df.columns.str.strip().str.lower()

# ✅ Ensure required columns exist
required_columns = {"isic_id", "benign_malignant"}
assert required_columns.issubset(train_df.columns), "❌ Missing columns in train_df"
assert required_columns.issubset(val_df.columns), "❌ Missing columns in val_df"

# ✅ Define image directories
IMAGE_DIR = "./train-images"
VAL_IMAGE_DIR = "./val-images"


# ✅ Function to check if image exists
def check_image_exists(isic_id, image_dir):
    for ext in [".jpg", ".jpeg", ".png"]:
        img_path = Path(image_dir) / f"{isic_id}{ext}"
        if img_path.is_file():
            return str(img_path)
    return None


train_df["image_path"] = train_df["isic_id"].apply(
    lambda x: check_image_exists(x, IMAGE_DIR)
)
val_df["image_path"] = val_df["isic_id"].apply(
    lambda x: check_image_exists(x, VAL_IMAGE_DIR)
)

# ✅ Drop rows where images are missing
train_df.dropna(subset=["image_path"], inplace=True)
val_df.dropna(subset=["image_path"], inplace=True)

print(f"🔍 Train dataset size after cleaning: {len(train_df)}")
print(f"🔍 Validation dataset size after cleaning: {len(val_df)}")

# ✅ Convert labels: Benign = 0, Malignant = 1
label_map = {"benign": 0, "malignant": 1}
train_df["benign_malignant"] = train_df["benign_malignant"].map(label_map).astype(int)
val_df["benign_malignant"] = val_df["benign_malignant"].map(label_map).astype(int)

# ✅ Compute Class Weights
unique_labels = np.unique(train_df["benign_malignant"])
class_weights = compute_class_weight(
    class_weight="balanced", classes=unique_labels, y=train_df["benign_malignant"]
)
class_weight_dict = {0: 0.6, 1: 3.0}  # Increase weight for malignant class

print(f"🔍 Computed Class Weights: {class_weight_dict}")


# ✅ Cosine Decay callback (to simulate CosineDecayRestarts)
def cosine_decay(epoch, lr_start=1e-5, lr_min=1e-6, T_max=30):
    from math import pi, cos

    return lr_min + 0.5 * (lr_start - lr_min) * (1 + cos(pi * (epoch % T_max) / T_max))


lr_callback = LearningRateScheduler(cosine_decay, verbose=1)

# ✅ Albumentations Augmentation (Optimized for Clear, Natural Lesions)
IMG_SIZE = (384, 384)
train_augmentation = A.Compose(
    [
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        A.HorizontalFlip(p=0.7),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.Affine(rotate=(-20, 20), shear=(-8, 8), scale=(0.95, 1.08), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.15, p=0.4),
        A.ElasticTransform(alpha=0.5, sigma=15, alpha_affine=15, p=0.2),
        A.GridDistortion(num_steps=5, distort_limit=0.07, p=0.3),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5),
        A.Sharpen(alpha=(0.3, 0.6), p=0.5),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
    ]
)

val_augmentation = A.Compose(
    [
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    ]
)


def load_and_augment_image(img_path, label, is_train=True):
    def process_image(img_path):
        img_path = img_path.decode("utf-8")  # Ensure string format for debugging
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)  # Resize image to target size
        img = tf.cast(
            img, tf.uint8
        ).numpy()  # Convert to NumPy array for Albumentations
        aug_pipeline = train_augmentation if is_train else val_augmentation
        augmented = aug_pipeline(image=img)["image"]
        return augmented.astype(np.float32) / 255.0  # Normalize image

    img = tf.numpy_function(func=process_image, inp=[img_path], Tout=tf.float32)
    img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])  # Set static shape for the image
    label = tf.cast(label, tf.float32)

    return img, label


def prepare_dataset(df, batch_size=32, shuffle=False, is_train=True):
    if "image_path" not in df.columns:
        raise KeyError("'image_path' column is missing in the DataFrame")

    df["image_path"] = df["image_path"].apply(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
    )

    print(df["image_path"].head())  # Debugging step: Print image paths to verify

    dataset = tf.data.Dataset.from_tensor_slices(
        (df["image_path"].values, df["benign_malignant"].values)
    )

    if shuffle:
        dataset = dataset.shuffle(len(df))

    dataset = dataset.map(
        lambda x, y: load_and_augment_image(x, y, is_train),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ✅ Example usage
train_dataset = prepare_dataset(train_df, batch_size=32, shuffle=True, is_train=True)
val_dataset = prepare_dataset(val_df, batch_size=32, is_train=False)

# ✅ Model Training - Add your model building and fitting here, e.g., using the build_model function:
# model = build_model(input_shape=(384, 384, 3))
# model.compile(optimizer=AdamW(), loss="binary_crossentropy", metrics=["accuracy", Recall(), Precision(), AUC()])
# model.fit(train_dataset, validation_data=val_dataset, epochs=30, class_weight=class_weight_dict, callbacks=[lr_callback, ModelCheckpoint, EarlyStopping])
# ✅ Load previously trained model
# model = build_model(img_size=(384, 384, 3), dropout_rate=0.4)

# Load the final model from the saved path
final_model_path = "./final_model.h5"  # Path to the final model
model = load_model(final_model_path)

model.summary()

# ✅ Unfreeze top layers of EfficientNet-B0 (starting from the last 40 layers)
base_model = model.get_layer("efficientnetb0")  # Get base model

# Unfreeze top layers (e.g., last 40 layers)
for layer in base_model.layers[-40:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

# ✅ Scalar LR for AdamW
optimizer = AdamW(learning_rate=1e-5, weight_decay=1e-5)

# ✅ Compile Model (with fine-tuning optimizer)
model.compile(
    optimizer=optimizer,
    loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.3, gamma=2.0),
    metrics=["accuracy", "AUC"],
)

# ✅ Callbacks for fine-tuning
fine_tune_callbacks = [
    ModelCheckpoint(
        "./final_model.h5",
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=2, verbose=1, min_lr=1e-7
    ),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
]

# ✅ Continue Training for Fine-Tuning (e.g., 20 more epochs)
class_weight_dict = {0: 0.6, 1: 3.0}
fine_tune_history = model.fit(
    train_dataset,
    epochs=10,  # Adjusted epochs (you can increase to 20 for more fine-tuning)
    validation_data=val_dataset,
    callbacks=fine_tune_callbacks,
    class_weight=class_weight_dict,
)

# ✅ Save the Model
model.save("./complete_model.h5")
print("✅ Model saved successfully.")

# ✅ Evaluate and Plot Results
eval_results = model.evaluate(val_dataset)
print(f"Evaluation Results after Fine-Tuning: {eval_results}")

# ✅ Predict raw probabilities (before thresholding)
y_prob = model.predict(val_dataset, verbose=1)

# ✅ Print predicted probabilities
print("🔍 Predicted probabilities for each image in validation set:")
for idx, prob in enumerate(y_prob):
    print(f"Image {idx + 1}: Malignant Probability = {prob[0]:.4f}")

# Optionally, print classification metrics
y_true = val_df["benign_malignant"].values
y_pred = (y_prob > 0.41).astype("int32")
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# Classification Report and Confusion Matrix
print("Classification Report:\n", classification_report(y_true, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("AUC Score:", roc_auc_score(y_true, y_pred))

# ✅ Plot Accuracy & Loss during Fine-Tuning
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(fine_tune_history.history["accuracy"], label="Train Accuracy")
plt.plot(fine_tune_history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Model Accuracy")

plt.subplot(1, 2, 2)
plt.plot(fine_tune_history.history["loss"], label="Train Loss")
plt.plot(fine_tune_history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Model Loss")

plt.show()

# # ✅ Precision-Recall AUC Calculation
# precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_prob)
# pr_auc_score = auc(recall_vals, precision_vals)
# print(f"🔬 Precision-Recall AUC: {pr_auc_score:.4f}")

# # ✅ Auto-tune threshold (from 0.2 to 0.5) to optimize F1-Score
# best_f1 = 0
# best_thresh = 0.3  # Start with the initial threshold of 0.3
# for thresh in np.arange(0.3, 0.51, 0.01):
#     y_pred = (y_prob > thresh).astype("int32")
#     f1 = f1_score(y_true, y_pred)
#     if f1 > best_f1:
#         best_f1 = f1
#         best_thresh = thresh
# print(f"🎯 Best threshold (based on F1): {best_thresh:.2f}, F1-Score: {best_f1:.4f}")

# # ✅ Final predictions using the best threshold
# y_pred_final = (y_prob > best_thresh).astype("int32")
# precision = precision_score(y_true, y_pred_final)
# recall = recall_score(y_true, y_pred_final)
# auc_score = roc_auc_score(y_true, y_pred_final)

# print(f"🧪 Final Metrics using Threshold {best_thresh:.2f}:")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1-Score: {best_f1:.4f}")
# print(f"AUC Score: {auc_score:.4f}")
# print("Classification Report:\n", classification_report(y_true, y_pred_final))
# print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred_final))

# # ✅ Plot Accuracy & Loss after Fine-Tuning
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(fine_tune_history.history["accuracy"], label="Train Accuracy")
# plt.plot(fine_tune_history.history["val_accuracy"], label="Validation Accuracy")
# plt.title("Model Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(fine_tune_history.history["loss"], label="Train Loss")
# plt.plot(fine_tune_history.history["val_loss"], label="Validation Loss")
# plt.title("Model Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()

# plt.tight_layout()
# plt.show()

# # ✅ Optional: Plot Precision-Recall Curve
# plt.figure(figsize=(6, 6))
# plt.plot(recall_vals, precision_vals, marker=".", label=f"PR AUC = {pr_auc_score:.4f}")
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Precision-Recall Curve")
# plt.legend()
# plt.grid(True)
# plt.show()
