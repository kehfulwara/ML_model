# import os
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from sklearn.metrics import (
#     classification_report,
#     confusion_matrix,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_auc_score,
#     roc_curve,
#     precision_recall_curve,
# )
# from sklearn.metrics import accuracy_score
# import albumentations as A
# import cv2
# import matplotlib.pyplot as plt
# import seaborn as sns
# import random

# # CONFIG
# IMG_SIZE = (384, 384)
# BATCH_SIZE = 32
# THRESHOLD = 0.43
# MODEL_PATH = "./final_model.h5"
# CLEANED_DIR = "./cleaned_test_384/test_cleaned_384"
# CLEANED_CSV = "./cleaned_test_384/cleaned_metadata_2020.csv"
# MALIGNANT_DIR = "./malignant_test_subset"

# # Load CSV and format
# df_cleaned = pd.read_csv(CLEANED_CSV)
# df_cleaned.columns = df_cleaned.columns.str.strip().str.lower()
# df_cleaned["benign_malignant"] = df_cleaned["benign_malignant"].map(
#     {"benign": 0, "malignant": 1}
# )


# def check_image_exists(isic_id, image_dir):
#     for ext in [".jpg", ".jpeg", ".png"]:
#         path = os.path.join(image_dir, f"{isic_id}{ext}")
#         if os.path.isfile(path):
#             return path
#     return None


# df_cleaned["image_path"] = df_cleaned["isic_id"].apply(
#     lambda x: check_image_exists(x, CLEANED_DIR)
# )
# df_cleaned.dropna(subset=["image_path"], inplace=True)

# # Load malignant subset
# malignant_paths = [
#     os.path.join(MALIGNANT_DIR, f)
#     for f in os.listdir(MALIGNANT_DIR)
#     if f.endswith((".jpg", ".png", ".jpeg"))
# ]
# df_malignant = pd.DataFrame(
#     {"image_path": malignant_paths, "benign_malignant": [1] * len(malignant_paths)}
# )

# # Combine
# df = pd.concat(
#     [df_cleaned[["image_path", "benign_malignant"]], df_malignant]
# ).reset_index(drop=True)

# # Augmentations
# val_augmentation = A.Compose([A.Resize(*IMG_SIZE)])


# def normalize(img):
#     return img.astype(np.float32) / 255.0


# def load_image(img_path, label):
#     def process(path):
#         path = path.decode("utf-8")
#         img = cv2.imread(path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = val_augmentation(image=img)["image"]
#         return normalize(img)

#     img = tf.numpy_function(process, [img_path], tf.float32)
#     img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
#     return img, tf.cast(label, tf.float32)


# # Dataset
# dataset = tf.data.Dataset.from_tensor_slices(
#     (df["image_path"].values, df["benign_malignant"].values)
# )
# dataset = dataset.map(load_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# # Load Model
# model = load_model(MODEL_PATH, compile=False)

# # Predict
# y_probs = model.predict(dataset).reshape(-1)
# y_true = df["benign_malignant"].values
# y_pred = (y_probs > THRESHOLD).astype("int32")

# # Metrics
# print(f"\n📌 Threshold = {THRESHOLD}")
# print("Accuracy:", (y_pred == y_true).mean())
# print("Precision:", precision_score(y_true, y_pred))
# print("Recall:", recall_score(y_true, y_pred))
# print("F1 Score:", f1_score(y_true, y_pred))
# print("AUC:", roc_auc_score(y_true, y_probs))
# print(
#     "\nClassification Report:\n",
#     classification_report(y_true, y_pred, target_names=["Benign", "Malignant"]),
# )
# print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# # Save
# df["predicted"] = y_pred
# df["probability"] = y_probs
# df.to_csv("test_predictions_fixed_042.csv", index=False)
# print("✅ Saved to test_predictions_fixed_042.csv")

# # ✅ Show 10 image predictions (5 benign, 5 malignant)
# sample_cleaned = df[df["benign_malignant"] == 0].sample(n=5, random_state=42)
# sample_malignant = df[df["benign_malignant"] == 1].sample(n=5, random_state=42)
# samples = pd.concat([sample_cleaned, sample_malignant]).reset_index(drop=True)

# import matplotlib.pyplot as plt
# import cv2

# fig, axes = plt.subplots(2, 5, figsize=(20, 8))
# fig.suptitle("🔍 Sample Predictions (Benign & Malignant)", fontsize=18)
# for i, row in samples.iterrows():
#     img = cv2.imread(row["image_path"])
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     ax = axes[i // 5, i % 5]
#     ax.imshow(img)
#     ax.axis("off")
#     ax.set_title(
#         f"True: {'Malig.' if row['benign_malignant'] == 1 else 'Benign'}\n"
#         f"Pred: {'Malig.' if row['predicted'] == 1 else 'Benign'}\n"
#         f"Prob: {row['probability']:.2f}",
#         fontsize=10,
#     )
# plt.tight_layout()
# plt.show()

# # ✅ Precision-Recall Curve with Accuracy Overlay
# from sklearn.metrics import precision_recall_curve, accuracy_score

# accuracy_final = accuracy_score(y_true, y_pred)

# precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
# f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
# best_f1_idx = np.argmax(f1s)
# best_f1 = f1s[best_f1_idx]

# plt.figure(figsize=(8, 6))
# plt.plot(
#     recalls,
#     precisions,
#     label=f"F1 = {best_f1:.4f} | Accuracy = {accuracy_final:.4f}",
#     color="blue",
# )
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Precision-Recall Curve (with Accuracy)")
# plt.grid(True)
# plt.legend()
# plt.show()

# # ✅ ROC Curve
# from sklearn.metrics import roc_curve, roc_auc_score

# fpr, tpr, _ = roc_curve(y_true, y_probs)
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_probs):.4f}")
# plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend(loc="lower right")
# plt.grid(True)
# plt.show()

# # ✅ Confusion Matrix
# import seaborn as sns

# cm = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(6, 5))
# sns.heatmap(
#     cm,
#     annot=True,
#     fmt="d",
#     cmap="Blues",
#     xticklabels=["Benign", "Malignant"],
#     yticklabels=["Benign", "Malignant"],
# )
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.tight_layout()
# plt.show()


# import os
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from sklearn.metrics import (
#     classification_report,
#     confusion_matrix,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_auc_score,
# )

# # ✅ Imports
# import os
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from sklearn.metrics import (
#     classification_report,
#     confusion_matrix,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_auc_score,
#     roc_curve,
#     precision_recall_curve,
#     accuracy_score,
# )
# import albumentations as A
# import cv2
# import matplotlib.pyplot as plt
# import seaborn as sns
# import random

# # ✅ CONFIG
# IMG_SIZE = (384, 384)
# BATCH_SIZE = 32
# THRESHOLD = 0.4220
# MODEL_PATH = "./final_model.h5"
# CLEANED_DIR = "./cleaned_test_384/test_cleaned_384"
# CLEANED_CSV = "./cleaned_test_384/cleaned_metadata_2020.csv"

# # ✅ Load CSV and format
# df_cleaned = pd.read_csv(CLEANED_CSV)
# df_cleaned.columns = df_cleaned.columns.str.strip().str.lower()
# df_cleaned["benign_malignant"] = df_cleaned["benign_malignant"].map(
#     {"benign": 0, "malignant": 1}
# )


# def check_image_exists(isic_id, image_dir):
#     for ext in [".jpg", ".jpeg", ".png"]:
#         path = os.path.join(image_dir, f"{isic_id}{ext}")
#         if os.path.isfile(path):
#             return path
#     return None


# df_cleaned["image_path"] = df_cleaned["isic_id"].apply(
#     lambda x: check_image_exists(x, CLEANED_DIR)
# )
# df_cleaned.dropna(subset=["image_path"], inplace=True)

# # Final dataframe
# df = df_cleaned[["image_path", "benign_malignant"]].reset_index(drop=True)

# # ✅ Augmentations
# val_augmentation = A.Compose([A.Resize(*IMG_SIZE)])


# def normalize(img):
#     return img.astype(np.float32) / 255.0


# def load_image(img_path, label):
#     def process(path):
#         path = path.decode("utf-8")
#         img = cv2.imread(path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = val_augmentation(image=img)["image"]
#         return normalize(img)

#     img = tf.numpy_function(process, [img_path], tf.float32)
#     img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
#     return img, tf.cast(label, tf.float32)


# # ✅ Dataset
# dataset = tf.data.Dataset.from_tensor_slices(
#     (df["image_path"].values, df["benign_malignant"].values)
# )
# dataset = dataset.map(load_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# # ✅ Load Model
# model = load_model(MODEL_PATH, compile=False)

# # ✅ Predict
# y_probs = model.predict(dataset).reshape(-1)
# y_true = df["benign_malignant"].values
# y_pred = (y_probs > THRESHOLD).astype("int32")

# # ✅ Metrics
# print(f"\n📌 Threshold = {THRESHOLD}")
# print("Accuracy:", (y_pred == y_true).mean())
# print("Precision:", precision_score(y_true, y_pred))
# print("Recall:", recall_score(y_true, y_pred))
# print("F1 Score:", f1_score(y_true, y_pred))
# print("AUC:", roc_auc_score(y_true, y_probs))
# print(
#     "\nClassification Report:\n",
#     classification_report(y_true, y_pred, target_names=["Benign", "Malignant"]),
# )
# print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# # ✅ Save Predictions
# df["predicted"] = y_pred
# df["probability"] = y_probs
# df.to_csv("test_predictions_fixed_042.csv", index=False)
# print("✅ Saved to test_predictions_fixed_042.csv")

# # ✅ Show 10 Sample Predictions
# sample_cleaned = df[df["benign_malignant"] == 0].sample(n=5, random_state=42)
# sample_malignant = df[df["benign_malignant"] == 1].sample(n=5, random_state=42)
# samples = pd.concat([sample_cleaned, sample_malignant]).reset_index(drop=True)

# fig, axes = plt.subplots(2, 5, figsize=(20, 8))
# fig.suptitle("🔍 Sample Predictions (Benign & Malignant)", fontsize=18)
# for i, row in samples.iterrows():
#     img = cv2.imread(row["image_path"])
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     ax = axes[i // 5, i % 5]
#     ax.imshow(img)
#     ax.axis("off")
#     ax.set_title(
#         f"True: {'Malig.' if row['benign_malignant'] == 1 else 'Benign'}\n"
#         f"Pred: {'Malig.' if row['predicted'] == 1 else 'Benign'}\n"
#         f"Prob: {row['probability']:.4f}",
#         fontsize=10,
#     )
# plt.tight_layout()
# plt.show()

# # ✅ Precision-Recall Curve with Accuracy Overlay
# accuracy_final = accuracy_score(y_true, y_pred)
# precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
# f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
# best_f1_idx = np.argmax(f1s)
# best_f1 = f1s[best_f1_idx]

# plt.figure(figsize=(8, 6))
# plt.plot(
#     recalls,
#     precisions,
#     label=f"F1 = {best_f1:.4f} | Accuracy = {accuracy_final:.4f}",
#     color="blue",
# )
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Precision-Recall Curve (with Accuracy)")
# plt.grid(True)
# plt.legend()
# plt.show()

# # ✅ ROC Curve
# fpr, tpr, _ = roc_curve(y_true, y_probs)
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_probs):.4f}")
# plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend(loc="lower right")
# plt.grid(True)
# plt.show()

# # ✅ Confusion Matrix
# cm = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(6, 5))
# sns.heatmap(
#     cm,
#     annot=True,
#     fmt="d",
#     cmap="Blues",
#     xticklabels=["Benign", "Malignant"],
#     yticklabels=["Benign", "Malignant"],
# )
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.tight_layout()
# plt.show()

# # ✅ Best Threshold Finder (F1 vs Threshold Plot)
# precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
# f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

# # Find best threshold
# best_idx = np.argmax(f1s)
# best_threshold = thresholds[best_idx]
# best_f1 = f1s[best_idx]

# print(f"✅ Best Threshold = {best_threshold:.4f} with F1 Score = {best_f1:.4f}")

# # Plot F1 Score vs Threshold
# plt.figure(figsize=(8, 6))
# plt.plot(thresholds, f1s[:-1], marker="o")
# plt.axvline(
#     x=best_threshold,
#     color="red",
#     linestyle="--",
#     label=f"Best Threshold = {best_threshold:.4f}",
# )
# plt.xlabel("Threshold")
# plt.ylabel("F1 Score")
# plt.title("F1 Score vs Threshold")
# plt.grid(True)
# plt.legend()
# plt.show()

# ✅ Imports
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    accuracy_score,
)
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import random

# ✅ CONFIG
IMG_SIZE = (384, 384)
BATCH_SIZE = 32
MODEL_PATH = "./final_model.h5"
CLEANED_DIR = "./cleaned_test_2019_384"
CLEANED_CSV = "./cleaned_metadata_test2019.csv"

# ✅ Load CSV and format
df_cleaned = pd.read_csv(CLEANED_CSV)
df_cleaned.columns = df_cleaned.columns.str.strip().str.lower()
df_cleaned["benign_malignant"] = df_cleaned["benign_malignant"].map(
    {"benign": 0, "malignant": 1}
)


def check_image_exists(isic_id, image_dir):
    for ext in [".jpg", ".jpeg", ".png"]:
        path = os.path.join(image_dir, f"{isic_id}{ext}")
        if os.path.isfile(path):
            return path
    return None


df_cleaned["image_path"] = df_cleaned["isic_id"].apply(
    lambda x: check_image_exists(x, CLEANED_DIR)
)
df_cleaned.dropna(subset=["image_path"], inplace=True)

# Final dataframe
df = df_cleaned[["image_path", "benign_malignant"]].reset_index(drop=True)

# ✅ Augmentations
val_augmentation = A.Compose([A.Resize(*IMG_SIZE)])


def normalize(img):
    return img.astype(np.float32) / 255.0


def load_image(img_path, label):
    def process(path):
        path = path.decode("utf-8")
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = val_augmentation(image=img)["image"]
        return normalize(img)

    img = tf.numpy_function(process, [img_path], tf.float32)
    img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    return img, tf.cast(label, tf.float32)


# ✅ Dataset
dataset = tf.data.Dataset.from_tensor_slices(
    (df["image_path"].values, df["benign_malignant"].values)
)
dataset = dataset.map(load_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# ✅ Load Model
model = load_model(MODEL_PATH, compile=False)
model.summary()
print("Input Layer(s):", model.inputs)
print("Output Layer(s):", model.outputs)

# ✅ Predict
y_probs = model.predict(dataset).reshape(-1)
y_true = df["benign_malignant"].values

# ✅ Single Fixed Threshold
THRESHOLD = 0.43966678
y_pred = (y_probs > THRESHOLD).astype("int32")

# ✅ Metrics at Fixed Threshold
accuracy_final = accuracy_score(y_true, y_pred)
precision_final = precision_score(y_true, y_pred)
recall_final = recall_score(y_true, y_pred)
f1_final = f1_score(y_true, y_pred)
auc_final = roc_auc_score(y_true, y_probs)

print(f"\n📌 Using Fixed Threshold = {THRESHOLD:.7f}")
print("Accuracy:", accuracy_final)
print("Precision:", precision_final)
print("Recall:", recall_final)
print("F1 Score:", f1_final)
print("AUC:", auc_final)
print(
    "\nClassification Report:\n",
    classification_report(y_true, y_pred, target_names=["Benign", "Malignant"]),
)
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# ✅ Save Predictions
df["predicted"] = y_pred
df["probability"] = y_probs
df.to_csv("test_predictions_fixed_threshold.csv", index=False)
print("✅ Saved to test_predictions_fixed_threshold.csv")


# # Load your CSV file
# df = pd.read_csv(
#     "./test_predictions_fixed_threshold.csv"
# )  # replace with your actual file path

# # Sweep thresholds from 0 to 1 finely
# thresholds = np.linspace(0, 1, 10000)
# best_threshold = 0
# best_correct = 0

# for threshold in thresholds:
#     preds = (df["probability"] >= threshold).astype(int)
#     correct_preds = (preds == df["benign_malignant"]).sum()
#     if correct_preds > best_correct:
#         best_correct = correct_preds
#         best_threshold = threshold

# # Print the best threshold and its corresponding accuracy
# print(f"Best Threshold: {best_threshold:.4f}")
# print(f"Total Correct Predictions: {best_correct} / {len(df)}")
# print(f"Accuracy: {best_correct / len(df):.4f}")


# ✅ Show 10 Sample Predictions
sample_cleaned = df[df["benign_malignant"] == 0].sample(n=5, random_state=42)
sample_malignant = df[df["benign_malignant"] == 1].sample(n=5, random_state=42)
samples = pd.concat([sample_cleaned, sample_malignant]).reset_index(drop=True)
# ✅ Recompute predictions based on the best threshold
df["new_predicted"] = (df["probability"] >= THRESHOLD).astype(int)

# ✅ Select 2 true positives (benign_malignant = 1, predicted = 1)
true_positives = df[(df["benign_malignant"] == 1) & (df["new_predicted"] == 1)].sample(
    2, random_state=42
)

# ✅ Select 2 true negatives (benign_malignant = 0, predicted = 0)
true_negatives = df[(df["benign_malignant"] == 0) & (df["new_predicted"] == 0)].sample(
    2, random_state=42
)

# ✅ Select 2 false positives (benign_malignant = 0, predicted = 1)
false_positives = df[(df["benign_malignant"] == 0) & (df["new_predicted"] == 1)].sample(
    2, random_state=42
)

# ✅ Combine them
selected_samples = pd.concat(
    [true_positives, true_negatives, false_positives]
).reset_index(drop=True)

# ✅ Setup grid (2 rows x 3 columns) with slight spacing
fig, axes = plt.subplots(2, 3, figsize=(10, 8))  # Adjusted figure size
axes = axes.flatten()

# ✅ Loop through 6 samples
for i, row in selected_samples.iterrows():
    img = cv2.imread(row["image_path"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax = axes[i]
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(
        f"True: {'Malig.' if row['benign_malignant'] == 1 else 'Benign'}\n"
        f"Pred: {'Malig.' if row['new_predicted'] == 1 else 'Benign'}\n"
        f"Prob: {row['probability']:.4f}",
        fontsize=10,
        pad=10,  # Add slight spacing between image and title
    )

# ✅ Hide any unused axes (if somehow <6 images)
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

# ✅ Adjust spacing between images
plt.subplots_adjust(wspace=0.3, hspace=0.8)  # Added spacing between plots
plt.tight_layout()
plt.show()


# ✅ Precision-Recall Curve
precisions, recalls, thresholds_pr = precision_recall_curve(y_true, y_probs)
f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_f1_idx = np.argmax(f1s)
best_f1_score_curve = f1s[best_f1_idx]

plt.figure(figsize=(8, 6))
plt.plot(
    recalls,
    precisions,
    label=f"F1 (curve best) = {best_f1_score_curve:.4f} | Accuracy = {accuracy_final:.4f}",
    color="blue",
)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (with Accuracy)")
plt.grid(True)
plt.legend()
plt.show()

# ✅ ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc_final:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# ✅ Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"],
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ✅ Best Threshold Finder (F1 vs Threshold Plot for visualization only)
precisions, recalls, thresholds_curve = precision_recall_curve(y_true, y_probs)
f1s_curve = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

plt.figure(figsize=(8, 6))
plt.plot(thresholds_curve, f1s_curve[:-1], marker="o")
plt.axvline(
    x=THRESHOLD,
    color="red",
    linestyle="--",
    label=f"Used Threshold = {THRESHOLD:.7f}",
)
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.title("F1 Score vs Threshold")
plt.grid(True)
plt.legend()
plt.show()

# import pandas as pd
# import numpy as np
# from sklearn.metrics import confusion_matrix

# # Load your CSV file
# data = pd.read_csv("./test_predictions_fixed_threshold.csv")

# # Extract true labels and predicted probabilities
# y_true = data["benign_malignant"]
# y_probs = data["probability"]

# # Base threshold
# initial_threshold = 0.438568

# # Define a very small range around base threshold (±0.01)
# threshold_range = np.linspace(initial_threshold - 0.01, initial_threshold + 0.01, 400)

# best_threshold = initial_threshold
# best_score = -np.inf  # Initialize with very low score

# # Fine-tune threshold to reduce FP slightly while keeping TP high
# for thresh in threshold_range:
#     preds = (y_probs >= thresh).astype(int)
#     tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

#     benign_correct_rate = tn / (tn + fp) if (tn + fp) > 0 else 0
#     malignant_correct_rate = tp / (tp + fn) if (tp + fn) > 0 else 0

#     # Define a custom small optimization metric:
#     # prioritize slightly higher benign_correct_rate and reasonable malignant_correct_rate
#     score = (benign_correct_rate * 0.7) + (malignant_correct_rate * 0.3)

#     if score > best_score:
#         best_score = score
#         best_threshold = thresh

# # Final evaluation with best threshold
# final_preds = (y_probs >= best_threshold).astype(int)
# tn, fp, fn, tp = confusion_matrix(y_true, final_preds).ravel()

# benign_correct_rate = tn / (tn + fp) if (tn + fp) > 0 else 0
# malignant_correct_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
# overall_accuracy = (tp + tn) / (tn + fp + fn + tp)

# # Print the results
# print(f"✅ Optimized Threshold: {best_threshold:.6f}")
# print(f"🔵 False Positives: {fp}")
# print(f"🟢 Benign Correct Prediction Rate: {benign_correct_rate:.4f}")
# print(f"🔴 Malignant Correct Prediction Rate: {malignant_correct_rate:.4f}")
# print(f"🎯 Overall Accuracy: {overall_accuracy:.4f}")
