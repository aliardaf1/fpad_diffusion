import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms, models
from PIL import Image

# 1. VERİ KÜMESİ (DATASET) SINIFI
class ReplayAttackDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        real_dir = os.path.join(root_dir, 'real')
        if os.path.exists(real_dir):
            for img_name in os.listdir(real_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(real_dir, img_name))
                    self.labels.append(1) # Gerçek = 1

        attack_dir = os.path.join(root_dir, 'attack')
        if os.path.exists(attack_dir):
            for img_name in os.listdir(attack_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(attack_dir, img_name))
                    self.labels.append(0) # Sahte = 0

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# 2. MODEL MİMARİSİ (RESNET-50)
def get_resnet50_classifier(device):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) 
    return model.to(device)

# 3. ÖN İŞLEME (PREPROCESS)
fpad_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
