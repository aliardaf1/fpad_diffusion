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
# 1. VERİ OKUMA SINIFLARI (OULU ÖZEL)
# ==========================================

# OULU hiyerarşik klasör yapısı için (os.walk ile derinlemesine arama)
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

# Tek bir klasördeki tüm resimlere aynı etiketi basmak için (Sentetik DDPM ve S2S verileri için)
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
def train_oulu_ultimate_augmented():
    # Kart Seçimi (Titan V veya 1080 Ti durumuna göre "cuda:0" veya "cuda:1" yapabilirsiniz)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"OULU ULTIMATE S2S Hibrit Eğitim Başlıyor... Cihaz: {device}")

    # --- YOLLAR ---
    ORIG_TRAIN_ROOT = "/home/undergrad25_1/Desktop/fpad_diffusion/data/processed/OULU/train"
    DEV_ROOT = "/home/undergrad25_1/Desktop/fpad_diffusion/data/processed/OULU/dev"
    
    # 1. Sentetik Veri Yolu (OULU DDPM 50K)
    DDPM_SPOOF_DIR = "/home/undergrad25_1/Desktop/fpad_diffusion/data/synthetic/OULU-NPU/DDPM_spoof/spoof"
    
    # 2. Sentetik Veri Yolu (OULU Spoof-to-Spoof 5K)
    S2S_SPOOF_DIR = "/home/undergrad25_1/Desktop/fpad_diffusion/data/S2S/OULU-NPU"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- VERİ SETLERİNİ BİRLEŞTİRME (3'LÜ KOMBİNASYON) ---
    print("OULU 3 Farklı Veri Seti Potada Birleştiriliyor...")
    
    # 1. Orijinal Eğitim Seti (Real + Attack)
    orig_train_ds = OuluDataset(ORIG_TRAIN_ROOT, transform=transform)
    
    # 2. Sentetik DDPM Sadece Attack Seti (Label: 0.0)
    ddpm_spoof_ds = SimpleFolderDataset(DDPM_SPOOF_DIR, label=0.0, transform=transform)

    # 3. Sentetik S2S Sadece Attack Seti (Label: 0.0)
    s2s_spoof_ds = SimpleFolderDataset(S2S_SPOOF_DIR, label=0.0, transform=transform)
    
    # ÜÇÜNÜ BİRLEŞTİRME
    full_train_ds = ConcatDataset([orig_train_ds, ddpm_spoof_ds, s2s_spoof_ds])
    
    # Doğrulama Seti (Orijinal kalmalı)
    dev_ds = OuluDataset(DEV_ROOT, transform=transform)

    print(f"OULU Orijinal Train: {len(orig_train_ds)} | OULU DDPM Attack: {len(ddpm_spoof_ds)} | OULU S2S Attack: {len(s2s_spoof_ds)}")
    print(f"Toplam Ultimate Eğitim Verisi: {len(full_train_ds)}")

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
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/20 [OULU Ultimate Eğitim]", mininterval=10)
        
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
        print(f"OULU DEV Metrikleri -> HTER: {hter*100:.2f}% | AUC: {auc:.4f} | APCER: {apcer*100:.2f}% | BPCER: {bpcer*100:.2f}%")

        # Nihai model kaydı
        if hter < best_hter:
            best_hter = hter
            torch.save(model.state_dict(), "resnet18_oulu_ultimate_s2s_best.pth")
            print(">> Yeni en iyi OULU ULTIMATE S2S modeli kaydedildi!")
        print("-" * 60)

if __name__ == "__main__":
    train_oulu_ultimate_augmented()
