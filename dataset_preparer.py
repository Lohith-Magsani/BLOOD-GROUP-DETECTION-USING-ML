import os
import shutil
import random
from tqdm import tqdm

# Path to the folder where fingerprint images are located
image_folder = 'SOCOFing/Real'

# Destination folder to store the dataset
dataset_folder = 'dataset'

# Blood group categories
categories = ['A', 'B', 'AB', 'O']

# Number of images to assign per category
images_per_category = 1000

def delete_existing_dataset():
    if os.path.exists(dataset_folder):
        shutil.rmtree(dataset_folder)
        print("[INFO] Existing dataset folder deleted.")

def create_folder_structure():
    os.makedirs(dataset_folder, exist_ok=True)
    for category in categories:
        os.makedirs(os.path.join(dataset_folder, category), exist_ok=True)
    print("[INFO] New dataset folder structure created.")

def move_images_to_folders():
    # Collect all image file paths
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.bmp'))]
    full_paths = [os.path.join(image_folder, f) for f in image_files]
    random.shuffle(full_paths)  # Shuffle for random distribution

    total_required = images_per_category * len(categories)

    if len(full_paths) < total_required:
        print(f"[ERROR] Not enough images in source folder. Required: {total_required}, Found: {len(full_paths)}")
        return

    # Assign images to each category equally
    index = 0
    for category in categories:
        category_folder = os.path.join(dataset_folder, category)
        for _ in tqdm(range(images_per_category), desc=f"Assigning to {category}"):
            image_path = full_paths[index]
            image_name = os.path.basename(image_path)
            destination_path = os.path.join(category_folder, image_name)
            shutil.copy(image_path, destination_path)
            index += 1
        print(f"[INFO] {images_per_category} images copied to '{category}' folder.")

def prepare_dataset():
    print("[INFO] Preparing dataset...")
    delete_existing_dataset()
    create_folder_structure()
    move_images_to_folders()
    print("[INFO] Dataset preparation complete.")

if __name__ == "__main__":
    prepare_dataset()
