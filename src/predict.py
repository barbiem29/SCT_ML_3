import joblib
import cv2
import os
import numpy as np

# 🔹 Load the trained model
model = joblib.load("../model/svm_model.pkl")

# 🔹 Image size and test directory
image_size = (64, 64)
test_dir = "../data/test"

# 🔹 Loop through test images
for file in os.listdir(test_dir):
    path = os.path.join(test_dir, file)
    img = cv2.imread(path)
    if img is not None:
        img = cv2.resize(img, image_size).flatten().reshape(1, -1)
        pred = model.predict(img)[0]
        label = "Dog" if pred == 1 else "Cat"
        print(f"{file}: {label}")
