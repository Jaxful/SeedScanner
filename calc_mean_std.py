import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Adjusted transformations to maintain aspect ratio and then crop
transform = transforms.Compose([
    transforms.Resize(256),  # Resize the smaller edge to 256 while maintaining aspect ratio
    transforms.CenterCrop(256),  # Crop the center to 256x256
    transforms.ToTensor(),
    # Do not apply normalization here since we're calculating mean and std
])

# Load datasets with transformations
train_dataset = ImageFolder(root='data/train', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

def calculate_mean_std(dataloader):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0

    for images, _ in dataloader:
        # Rearrange batch to be the shape of [B, C, W * H]
        images = images.view(images.size(0), images.size(1), -1)
        # Update total_images
        total_images += images.size(0)
        # Compute mean and std here
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    # Final calculation
    mean /= total_images
    std /= total_images

    return mean, std

# Calculate mean and std
mean, std = calculate_mean_std(train_loader)

print(f"Calculated Mean: {mean}")
print(f"Calculated Std: {std}")