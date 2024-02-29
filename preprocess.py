"""Data preprocessing module for dog breed classification."""
import os
import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class DogBreedDataset(Dataset):
    """Custom dataset for dog breed images."""
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = cv2.imread(img_path)
        if self.transform:
            image = self.transform(image)
        return image, self.labels.iloc[idx, 1]


def get_transforms(train=True):
    """Get data augmentation transforms."""
    from torchvision import transforms
    if train:
        return transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
