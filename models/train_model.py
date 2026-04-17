# =============================================================================
# IMPROVED EFFICIENTNET TRAINING FOR SKIN LESIONS
# =============================================================================

import os
import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# ================= CONFIG =================

IMG_SIZE = 224
BATCH_SIZE = 16  # Increase to 32 if you have enough RAM
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 5

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
MODEL_SAVE_PATH = "outputs/models/final_model.h5"

os.makedirs("outputs/models", exist_ok=True)

# ================= DATA AUGMENTATION =================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

# ================= LOAD DATA =================

print("📦 Loading datasets...")

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=42
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

CLASS_NAMES = train_ds.class_names
NUM_CLASSES = len(CLASS_NAMES)

print("Classes:", CLASS_NAMES)
print(f"Number of training batches: {len(train_ds)}")
print(f"Number of validation batches: {len(val_ds)}")

# ================= PREPROCESS =================

def preprocess_train(x, y):
    # Apply data augmentation
    x = data_augmentation(x, training=True)
    # EfficientNet-specific preprocessing
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    return x, y

def preprocess_val(x, y):
    # EfficientNet-specific preprocessing (no augmentation)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    return x, y

train_ds = train_ds.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess_val, num_parallel_calls=tf.data.AUTOTUNE)

# Prefetch for performance
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# ================= CLASS WEIGHTS =================

print("⚖️ Calculating class weights...")

# Get labels from original dataset (before preprocessing)
train_ds_for_labels = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

labels = []
for _, y in train_ds_for_labels:
    labels.append(np.argmax(y.numpy(), axis=1))

labels = np.concatenate(labels)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

# Cap extreme weights to prevent instability
class_weights = np.clip(class_weights, 0.5, 10.0)

class_weights = dict(enumerate(class_weights))
print("Class weights (capped):", class_weights)

# Print class distribution
unique, counts = np.unique(labels, return_counts=True)
for i, (cls, count) in enumerate(zip(unique, counts)):
    print(f"  {CLASS_NAMES[cls]}: {count} samples (weight: {class_weights[i]:.2f})")

# ================= MODEL =================

print("🧠 Building EfficientNetB0 model...")

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False  # Freeze initially

# Build model
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = tf.keras.Model(inputs=inputs, outputs=output)

# ================= COMPILE =================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Lower initial LR
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")]
)

# ================= CALLBACKS =================

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "outputs/models/best_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]

# ================= TRAIN PHASE 1 (FROZEN) =================

print("\n🚀 Training Phase 1 (Frozen base, training top layers)...")
print("="*60)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_PHASE1,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ================= FINE-TUNING =================

print("\n🔓 Fine-tuning (unfreezing last 50 layers)...")
print("="*60)

# Unfreeze the last 50 layers
for layer in base_model.layers[-50:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

print(f"Trainable layers: {sum([1 for layer in model.layers if layer.trainable])}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Very low LR for fine-tuning
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")]
)

# ================= TRAIN PHASE 2 (FINE-TUNING) =================

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_PHASE2,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ================= EVALUATE =================

print("\n📊 Final Evaluation:")
print("="*60)

val_loss, val_acc, val_top3 = model.evaluate(val_ds)
print(f"Validation Accuracy: {val_acc*100:.2f}%")
print(f"Validation Top-3 Accuracy: {val_top3*100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")

# ================= SAVE MODEL =================

model.save(MODEL_SAVE_PATH)

print("\n✅ Training complete!")
print(f"📁 Model saved at: {MODEL_SAVE_PATH}")
print(f"📁 Best model saved at: outputs/models/best_model.h5")

# ================= TRAINING SUMMARY =================

print("\n📈 Training Summary:")
print("="*60)
print(f"Best validation accuracy: {max(history1.history['val_accuracy'] + history2.history['val_accuracy'])*100:.2f}%")
print(f"Final training accuracy: {history2.history['accuracy'][-1]*100:.2f}%")