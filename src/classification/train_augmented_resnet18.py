import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# ==========================================
# 1. VERİ OKUMA SINIFLARI
# ==========================================

# Orijinal Replay-Attack klasör yapısı için (real/ ve attack/ alt klasörleri olanlar)
class ReplayAttackDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Gerçekler (Label: 1.0)
        real_dir = os.path.join(root_dir, 'real')
        if os.path.exists(real_dir):
            for img in os.listdir(real_dir):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(real_dir, img))
                    self.labels.append(1.0)

        # Sahteler (Label: 0.0)
        attack_dir = os.path.join(root_dir, 'attack')
        if os.path.exists(attack_dir):
            for img in os.listdir(attack_dir):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(attack_dir, img))
                    self.labels.append(0.0)

    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)

# Tek bir klasördeki tüm resimlere aynı etiketi basmak için (Sentetik Attack verileri için)
class SimpleFolderDataset(Dataset):
    def __init__(self, root_dir, label, transform=None):
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.labels = [float(label)] * len(self.image_paths)
        self.transform = transform

    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)

# ==========================================
# 2. MİMARİ VE METRİKLER (PSD/ADD Uyumlu)
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
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.0
    return apcer, bpcer, hter, auc

# ==========================================
# 3. EĞİTİM DÖNGÜSÜ
# ==========================================
def train_augmented():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Hibrit Eğitim Başlıyor... Cihaz: {device}")

    # --- YOLLAR ---
    ORIG_TRAIN_ROOT = "/home/undergrad25_1/Desktop/fpad_diffusion/data/processed/ReplayAttack/train"
    SYNTH_ATTACK_DIR = "/home/undergrad25_1/Desktop/fpad_diffusion/data/synthetic/ReplayAttack/DDPM_spoof/spoof"
    DEV_ROOT = "/home/undergrad25_1/Desktop/fpad_diffusion/data/processed/ReplayAttack/dev"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- VERİ SETLERİNİ BİRLEŞTİRME ---
    print("Veri setleri hazırlanıyor...")
    
    # 1. Orijinal Eğitim Seti (Real + Attack)
    orig_train_ds = ReplayAttackDataset(ORIG_TRAIN_ROOT, transform=transform)
    
    # 2. Sentetik Sadece Attack Seti (Label: 0.0)
    synth_attack_ds = SimpleFolderDataset(SYNTH_ATTACK_DIR, label=0.0, transform=transform)
    
    # Birleştirme
    full_train_ds = ConcatDataset([orig_train_ds, synth_attack_ds])
    
    # Doğrulama Seti (Orijinal kalmalı)
    dev_ds = ReplayAttackDataset(DEV_ROOT, transform=transform)

    print(f"Orijinal Train: {len(orig_train_ds)} | Sentetik Attack: {len(synth_attack_ds)}")
    print(f"Toplam Eğitim Verisi: {len(full_train_ds)}")

    train_loader = DataLoader(full_train_ds, batch_size=32, shuffle=True, num_workers=4)
    dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, num_workers=4)

    # Model Hazırlığı
    model = get_resnet18_bce(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_hter = float('inf')
    for epoch in range(20):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/20 [Hibrit Eğitim]", mininterval=10)
        
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Doğrulama (Dev set üzerinde test)
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
        print(f"DEV Metrikleri -> HTER: {hter*100:.2f}% | AUC: {auc:.4f} | APCER: {apcer*100:.2f}% | BPCER: {bpcer*100:.2f}%")

        if hter < best_hter:
            best_hter = hter
            torch.save(model.state_dict(), "resnet18_augmented_final.pth")
            print(">> Yeni en iyi hibrit model kaydedildi!")
        print("-" * 60)

if __name__ == "__main__":
    train_augmented()
