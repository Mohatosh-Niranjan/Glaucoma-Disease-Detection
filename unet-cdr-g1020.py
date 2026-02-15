# unet-cdr-g1020.py
# U-Net for CDR estimation on G1020 dataset
# Fixed for CSV with columns: imageID, binaryLabels

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

# ----------------------------
# Configuration
# ----------------------------
DATA_ROOT = 'G1020'  # Path to G1020 folder
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 50

# Folder names
IMG_FOLDER = 'Images_Cropped/img'
MASK_FOLDER = 'Masks_Cropped/img'

# ----------------------------
# Load All Data from Root
# ----------------------------
def load_data(folder):
    images = []
    masks = []

    img_dir = os.path.join(folder, IMG_FOLDER)
    mask_dir = os.path.join(folder, MASK_FOLDER)

    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    for fname in sorted(os.listdir(img_dir)):
        if not fname.endswith('.jpg'):
            continue

        # Load image
        img_path = os.path.join(img_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to load image {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        images.append(img)

        # Load mask
        mask_name = fname.replace('.jpg', '.png')
        mask_path = os.path.join(mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Failed to load mask {mask_path}")
            continue
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        # Create binary masks: disc=1, cup=2
        disc_mask = (mask == 1).astype(np.float32)
        cup_mask = (mask == 2).astype(np.float32)
        stacked_mask = np.stack([disc_mask, cup_mask], axis=-1)
        masks.append(stacked_mask)

    return np.array(images), np.array(masks)

print("Loading all data from G1020...")
X_all, y_all = load_data(DATA_ROOT)

# ----------------------------
# Load CSV with Correct Column Names
# ----------------------------
csv_path = os.path.join(DATA_ROOT, 'G1020.csv')
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

df = pd.read_csv(csv_path)
print("CSV loaded. Columns:", df.columns.tolist())

# Use correct column names
filename_col = 'imageID'
label_col = 'binaryLabels'

if filename_col not in df.columns:
    raise ValueError(f"Column '{filename_col}' not found in CSV")
if label_col not in df.columns:
    raise ValueError(f"Column '{label_col}' not found in CSV")

filenames = df[filename_col].astype(str).values
labels = df[label_col].values


img_dir = os.path.join(DATA_ROOT, IMG_FOLDER)
image_files = [f for f in sorted(os.listdir(img_dir)) if f.endswith('.jpg')]
image_stems = [os.path.splitext(f)[0] for f in image_files]


matched_indices = []
for fname in filenames:
    stem = os.path.splitext(str(fname))[0]
    if stem in image_stems:
        idx = image_stems.index(stem)
        matched_indices.append(idx)
    else:
        print(f"Warning: Image '{fname}' not found in {IMG_FOLDER}")

if len(matched_indices) == 0:
    raise ValueError("No matching images found! Check file naming.")


X_all = X_all[matched_indices]
y_all = y_all[matched_indices]
labels = labels[:len(matched_indices)]

# Split into train/val (stratified by label)
train_idx, val_idx = train_test_split(
    np.arange(len(X_all)),
    test_size=0.2,
    stratify=labels,
    random_state=42
)

X_train = X_all[train_idx]
X_val = X_all[val_idx]
y_train = y_all[train_idx]
y_val = y_all[val_idx]

print(f"Train: {X_train.shape}, Val: {X_val.shape}")

# Split masks for dual output
y_train_disc = y_train[..., 0:1]
y_train_cup = y_train[..., 1:2]
y_val_disc = y_val[..., 0:1]
y_val_cup = y_val[..., 1:2]

# ----------------------------
# Dice Loss for Better Cup Segmentation
# ----------------------------
def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
    return 1.0 - dice

# ----------------------------
# U-Net Model
# ----------------------------
def create_unet():
    inputs = Input((IMG_SIZE, IMG_SIZE, 3))

    # Encoder
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(0.2)(p1)

    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(0.2)(p2)

    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(0.3)(p3)

    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(0.4)(p4)

    # Bottleneck
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(c5)
    c5 = Dropout(0.5)(c5)

    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(c6)
    c6 = Dropout(0.4)(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(c7)
    c7 = Dropout(0.3)(c7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(c8)
    c8 = Dropout(0.2)(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(c9)

    # Outputs
    disc_out = Conv2D(1, (1, 1), activation='sigmoid', name='disc')(c9)
    cup_out = Conv2D(1, (1, 1), activation='sigmoid', name='cup')(c9)

    return Model(inputs=inputs, outputs=[disc_out, cup_out])

# ----------------------------
# Build and Compile Model
# ----------------------------
model = create_unet()
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss={'disc': 'binary_crossentropy', 'cup': dice_loss},
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)
]

# ----------------------------
# Train
# ----------------------------
print("Starting training...")
history = model.fit(
    X_train,
    {'disc': y_train_disc, 'cup': y_train_cup},
    validation_data=(X_val, {'disc': y_val_disc, 'cup': y_val_cup}),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ----------------------------
# Save Model
# ----------------------------
model.save('unet_cdr_model.h5')
print("Model saved as 'unet_cdr_model.h5'")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('unet_cdr_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("Model saved as 'unet_cdr_model.tflite'")

# ----------------------------
# Post-Processing: Clean up masks
# ----------------------------
def postprocess_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

# ----------------------------
# Compute CDR Function
# ----------------------------
def compute_cdr(disc_mask, cup_mask):
    def get_diameter(mask):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return 0
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        return max(w, h)
    disc_diam = get_diameter(disc_mask)
    cup_diam = get_diameter(cup_mask)
    cdr = cup_diam / (disc_diam + 1e-6)
    return cdr, disc_diam, cup_diam

# ----------------------------
# Plot Results
# ----------------------------
def plot_results(image, true_disc, true_cup, pred_disc, pred_cup, cdr, title=""):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(true_disc, cmap='gray')
    plt.title("True Disc")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(true_cup, cmap='gray')
    plt.title("True Cup")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    overlay = (image * 255).astype(np.uint8).copy()
    overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    disc_overlay = np.dstack([pred_disc*255, np.zeros_like(pred_disc), np.zeros_like(pred_disc)]).astype(np.uint8)
    cup_overlay = np.dstack([np.zeros_like(pred_cup), np.zeros_like(pred_cup), pred_cup*255]).astype(np.uint8)
    overlay = cv2.addWeighted(overlay, 0.7, disc_overlay, 0.7, 0)
    overlay = cv2.addWeighted(overlay, 1.0, cup_overlay, 0.7, 0)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted (CDR: {cdr:.3f})")
    plt.axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

# ----------------------------
# Test on 5 Validation Images
# ----------------------------
print("\nTesting on 5 validation images...\n")
for i in range(5):
    idx = i
    img = X_val[idx:idx+1]
    true_disc = y_val_disc[idx].squeeze()
    true_cup = y_val_cup[idx].squeeze()

    pred_disc_out, pred_cup_out = model.predict(img)
    pred_disc = (pred_disc_out[0,:,:,0] > 0.5).astype(np.uint8)
    pred_cup = (pred_cup_out[0,:,:,0] > 0.5).astype(np.uint8)

    # Apply post-processing
    pred_disc = postprocess_mask(pred_disc)
    pred_cup = postprocess_mask(pred_cup)

    cdr, disc_d, cup_d = compute_cdr(pred_disc, pred_cup)

    plot_results(X_val[idx], true_disc, true_cup, pred_disc, pred_cup, cdr, f"Image {i+1} | CDR: {cdr:.3f}")
    print(f"Image {i+1}: Disc={disc_d:.1f}px, Cup={cup_d:.1f}px, CDR={cdr:.3f}")
    print(f"Glaucoma Risk: {'High' if cdr > 0.7 else 'Low'}\n")