import os
import cv2
import numpy as np
from tqdm import tqdm

def load_images(data_dir, image_size=(64, 64)):
    X, y = [], []
    categories = ['cats', 'dogs']

    for label, category in enumerate(categories):
        folder = os.path.join(data_dir, category)
        print(f"\n📂 Scanning folder: {folder}")

        for filename in tqdm(os.listdir(folder), desc=f"Loading {category}"):
            img_path = os.path.join(folder, filename)

            # Ignore non-image files
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"⚠️ Skipped non-image file: {filename}")
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"🚫 Failed to read: {img_path}")
                continue

            try:
                img = cv2.resize(img, image_size)
                X.append(img.flatten())
                y.append(label)
            except Exception as e:
                print(f"❌ Error resizing {img_path}: {e}")

    print(f"\n✅ Total images loaded: {len(X)}")
    return np.array(X), np.array(y)
