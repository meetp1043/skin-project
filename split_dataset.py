import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIRS = [
    "data/raw/HAM10000_images_part_1",
    "data/raw/HAM10000_images_part_2"
]

METADATA_PATH = "data/raw/HAM10000_metadata.csv"
OUTPUT_DIR = "data"

df = pd.read_csv(METADATA_PATH)
df["label"] = df["dx"]

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

# Build image lookup
image_map = {}
for d in RAW_DIRS:
    for file in os.listdir(d):
        if file.endswith(".jpg"):
            image_id = file.replace(".jpg", "")
            image_map[image_id] = os.path.join(d, file)

print(f"✅ Total images found: {len(image_map)}")

# Create folders
for split in ["train", "val"]:
    for label in df["label"].unique():
        os.makedirs(os.path.join(OUTPUT_DIR, split, label), exist_ok=True)

# Copy images
def copy_images(dataframe, split):
    for _, row in dataframe.iterrows():
        img_id = row["image_id"]
        label = row["label"]

        if img_id in image_map:
            src = image_map[img_id]
            dst = os.path.join(OUTPUT_DIR, split, label, img_id + ".jpg")
            shutil.copy(src, dst)
        else:
            print(f"❌ Missing: {img_id}")

print("📦 Copying training images...")
copy_images(train_df, "train")

print("📦 Copying validation images...")
copy_images(val_df, "val")

print("✅ Dataset split complete!")   