import os
from torchvision import transforms
from PIL import Image

def augment_images(directory, transformations, n_augmented_images):
    images = [img for img in os.listdir(directory) if img.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    for img_name in images:
        img_path = os.path.join(directory, img_name)
        img = Image.open(img_path).convert('RGB')
        
        # Generate augmented images
        for i in range(n_augmented_images):
            augmented_img = transformations(img)
            augmented_img_path = os.path.join(directory, f"aug_{i}_{img_name}")
            augmented_img.save(augmented_img_path)

# Define your transformations, focusing on those that preserve original content
transformations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    # Removed RandomRotation to avoid introducing black areas
])

# Directories for "Best" and "Moderate" images
best_dir = "./all_images/Best"
moderate_dir = "./all_images/Moderate"

# Specify the number of augmented images you want to create per original image
n_augmented_images = 3  # Adjust as needed

# Augment images in each category
augment_images(best_dir, transformations, n_augmented_images)
augment_images(moderate_dir, transformations, n_augmented_images)
