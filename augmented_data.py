import tensorflow as tf
import albumentations as A
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Target Image Size
IMG_SIZE = (384, 384)
TRAIN_IMG_DIR = "train-images"  # Ensure this path is correct

# ✅ **Refined Training Data Augmentation**

# Optimized Augmentation Pipeline

train_augmentation = A.Compose(
    [
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.3),
        A.Affine(
            translate_percent=(0.0, 0.02),
            scale=(0.95, 1.05),
            rotate=(-10, 10),
            shear=(-5, 5),
            p=0.2,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf(
            [
                A.GridDistortion(p=0.3),
                A.ElasticTransform(p=0.3),
                A.OpticalDistortion(p=0.3),
            ],
            p=0.3,
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


# Validation Augmentation (No Augmentations, Only Resize & Normalize)
val_augmentation = A.Compose(
    [
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),  # Only resize
        A.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),  # EfficientNet Normalization
    ]
)


def augment_image(image_input, is_train=True):
    """Applies augmentation to the image tensor."""

    # Ensure the image_input is not None
    if image_input is None:
        raise ValueError(
            "❌ Image input is None. Please provide a valid image path or image array."
        )

    # If input is a TensorFlow tensor (image is already loaded as tensor)
    if isinstance(image_input, tf.Tensor):
        # Convert the tensor to NumPy array only if eager execution is enabled
        if tf.executing_eagerly():  # Only convert if in eager execution mode
            image_input = image_input.numpy()
    else:
        # If input is a NumPy array, apply augmentation directly
        image_input = image_input

    # Choose the augmentation pipeline
    aug_pipeline = train_augmentation if is_train else val_augmentation

    # Apply augmentation pipeline (works with NumPy array now)
    augmented = aug_pipeline(image=image_input)["image"]

    # Convert back to TensorFlow tensor if necessary
    augmented = tf.convert_to_tensor(augmented, dtype=tf.float32)

    # Normalize image (convert to float32 and scale to [0,1])
    augmented = augmented / 255.0

    return augmented


# ✅ **Test the augmentation on a sample image**
if __name__ == "__main__":
    # 🔍 Find an image from the dataset
    image_files = [
        f
        for f in os.listdir(TRAIN_IMG_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_files:
        raise FileNotFoundError(
            f"❌ No image files found in {TRAIN_IMG_DIR}. Check dataset!"
        )

    img_path = os.path.join(TRAIN_IMG_DIR, image_files[0])  # ✅ Get the first image
    print(f"✅ Using sample image: {img_path}")

    # ✅ Load Image
    test_image = cv2.imread(img_path)
    if test_image is None:
        raise ValueError("❌ Image could not be read. Check file format!")

    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)  # Convert for augmentation
    aug_image = augment_image(test_image, is_train=True)

    # ✅ **Display Original vs Augmented Image**
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(test_image)
    ax[0].set_title("Original Image")

    # 🔍 Normalize for display
    aug_display = aug_image * np.array([0.229, 0.224, 0.225]) + np.array(
        [0.485, 0.456, 0.406]
    )
    aug_display = np.clip(aug_display, 0, 1)  # Ensure values are in [0,1]

    ax[1].imshow(aug_display)
    ax[1].set_title("Augmented Image (Optimized)")

    plt.show()

    # ✅ **Save Augmented Image**
    output_path = "augmented_data"  # Folder to save augmented images
    os.makedirs(output_path, exist_ok=True)  # Ensure directory exists

    # **Corrected Debugging for Augmented Image**
    if "aug_image" not in locals() and "aug_image" not in globals():
        print("❌ Error: aug_image is not defined.")
    elif aug_image is None:
        print("❌ Error: aug_image is None. Check augmentation pipeline.")
    else:
        # Convert RGB to BGR for OpenCV
        aug_image_bgr = cv2.cvtColor(
            (aug_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
        )

        # ✅ Extract filename dynamically
        filename = os.path.basename(img_path)  # Extracts "ISIC_0000000.jpg"
        save_path = os.path.join(output_path, filename)

        # ✅ Save the image
        success = cv2.imwrite(save_path, aug_image_bgr)

        if success:
            print(f"✅ Image saved successfully at: {save_path}")
        else:
            print("❌ Error: Failed to save the image. Check image format and data.")

    # ✅ Debugging Information
    print("Type of aug_image:", type(aug_image))
    print(
        "Shape of aug_image:", aug_image.shape if hasattr(aug_image, "shape") else "N/A"
    )
