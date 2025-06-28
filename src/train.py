from preprocess import load_images
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from collections import Counter
import joblib
import os
import numpy as np

# 🔹 Load images
X, y = load_images("../data/train", image_size=(64, 64))

# 🧾 Check original label distribution
print("🧾 Original Label Count:", Counter(y))

# ✅ Balance dataset: 1000 cats + 1000 dogs
cat_indices = np.where(np.array(y) == 0)[0][:1000]
dog_indices = np.where(np.array(y) == 1)[0][:1000]
selected_indices = np.concatenate([cat_indices, dog_indices])

X = np.array(X)[selected_indices]
y = np.array(y)[selected_indices]

print("🧾 Balanced Label Count:", Counter(y))

# 🔹 Split into train/val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Train SVM model
print("🧠 Training the model...")
model = SVC(kernel='linear', probability=True, max_iter=1000)
model.fit(X_train, y_train)
print("✅ Model training complete!")

# 🔹 Evaluate accuracy
preds = model.predict(X_val)
acc = accuracy_score(y_val, preds)
print(f"🎯 Validation Accuracy: {acc:.4f}")

# 🔹 Save model
# ✅ Save model to ../model/
os.makedirs("../model", exist_ok=True)
joblib.dump(model, "../model/svm_model.pkl")
print("💾 Model saved to: ../model/svm_model.pkl")
