import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dropout,
    Dense,
    BatchNormalization,
)
from tensorflow.keras.models import Model


def build_model(img_size, dropout_rate):
    # ✅ Load Pretrained EfficientNet-B0
    base_model = EfficientNetB0(
        include_top=False, input_shape=img_size, weights="imagenet"
    )

    # ✅ Freeze base layers initially
    base_model.trainable = False

    # ✅ Custom Classification Head
    inputs = tf.keras.Input(shape=img_size)
    x = base_model(inputs, training=False)  # ✅ Prevent BatchNorm issues
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)  # ✅ Added Batch Norm
    x = Dropout(dropout_rate)(x)  # ✅ Increased Dropout for better regularization
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)  # ✅ Binary Classification

    # ✅ Create Model
    model = Model(inputs=inputs, outputs=outputs)

    return model
