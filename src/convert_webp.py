from PIL import Image
import os

input_folder = "../data/test"
output_folder = "../data/test"

for filename in os.listdir(input_folder):
    if filename.endswith(".webp"):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert("RGB")
        new_filename = filename.replace(".webp", ".jpg")
        img.save(os.path.join(output_folder, new_filename), "JPEG")
        print(f"✅ Converted: {filename} → {new_filename}")
