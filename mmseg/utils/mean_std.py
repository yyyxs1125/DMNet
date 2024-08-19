import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Replace this with your dataset path
dataset = datasets.ImageFolder(root='/home2/lmfm45/mmsegmentation/data/my_dataset/split_image/', 
                               transform=transforms.ToTensor())

loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

mean = torch.zeros(3)
std = torch.zeros(3)
n_samples = 0

for images, _ in loader:
    # Flatten the image tensors
    images = images.view(images.size(0), images.size(1), -1)
    # Update the mean and std
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    n_samples += images.size(0)

mean /= n_samples
std /= n_samples

print(f"Mean: {mean}")
print(f"Std: {std}")
