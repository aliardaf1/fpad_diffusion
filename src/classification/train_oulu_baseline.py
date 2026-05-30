import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# ==========================================
# 1. OULU'YA ÖZEL VERİ OKUYUCU (Recursive/Derin Okuma)
# ==========================================
class OuluDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # Gerçekler (Label: 1.0)
        real_dir = os.path.join(root_dir, 'real')
        if os.path.exists(real_dir):
            for subdir, _, files in os.walk(real_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(subdir, file))
                        self.labels.append(1.0)

        # Sahteler (Label: 0.0)
        attack_dir = os.path.join(root_dir, 'attack')
        if os.path.exists(attack_dir):
            for subdir, _, files in os.walk(attack_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(subdir, file))
                        self.labels.append(0.0)

    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)

# ==========================================
# 2. MİMARİ VE METRİKLER
# ==========================================
def get_resnet18_bce(device):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1) 
    return model.to(device)

def calculate_fpad_metrics(y_true, y_prob):
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    TP = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    TN = sum((yt == 0 and yp == 0) for yt, yp in zip(y_true, y_pred))
    FP = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    FN = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))
    apcer = FP / (FP + TN + 1e-8)
    bpcer = FN / (FN + TP + 1e-8)
    hter = (apcer + bpcer) / 2.0
    try: auc = roc_auc_score(y_true, y_prob)
    except: auc = 0.0
    return apcer, bpcer, hter, auc

# ==========================================
# 3. EĞİTİM DÖNGÜSÜ
# ==========================================
def train_oulu_baseline():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"OULU Baseline Eğitimi Başlıyor... Cihaz: {device}")

    # --- OULU YOLLARI ---
    train_dir = "/home/undergrad25_1/Desktop/fpad_diffusion/data/processed/OULU/train"
    dev_dir = "/home/undergrad25_1/Desktop/fpad_diffusion/data/processed/OULU/dev"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("OULU Veri Setleri Yükleniyor...")
    train_dataset = OuluDataset(root_dir=train_dir, transform=transform)
    dev_dataset = OuluDataset(root_dir=dev_dir, transform=transform)

    print(f"OULU Train: {len(train_dataset)} | OULU Dev: {len(dev_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = get_resnet18_bce(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_hter = float('inf')
    for epoch in range(20):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/20 [OULU Baseline]", mininterval=10)
        
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in dev_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze(1)
                all_probs.extend(torch.sigmoid(outputs).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        apcer, bpcer, hter, auc = calculate_fpad_metrics(all_labels, all_probs)
        print(f"\nEpoch {epoch+1} Bitti | Loss: {running_loss/len(train_loader):.4f}")
        print(f"OULU DEV Metrikleri -> HTER: {hter*100:.2f}% | AUC: {auc:.4f} | APCER: {apcer*100:.2f}% | BPCER: {bpcer*100:.2f}%")

        if hter < best_hter:
            best_hter = hter
            torch.save(model.state_dict(), "resnet18_oulu_baseline_best.pth")
            print(">> Yeni en iyi OULU Baseline modeli kaydedildi!")
        print("-" * 60)

if __name__ == "__main__":
    train_oulu_baseline()
