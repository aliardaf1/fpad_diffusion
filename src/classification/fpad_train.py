import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score  # ADD Belgesi: AUC-ROC Metriği için

# ==========================================
# 1. VERİ KÜMESİ (DATASET) SINIFI
# ==========================================
class ReplayAttackDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Gerçek (Bona Fide) -> Etiket: 1.0 (BCE Loss için Float olmalı)
        real_dir = os.path.join(root_dir, 'real')
        if os.path.exists(real_dir):
            for img_name in os.listdir(real_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(real_dir, img_name))
                    self.labels.append(1.0)

        # Sahte (Spoof/Attack) -> Etiket: 0.0
        attack_dir = os.path.join(root_dir, 'attack')
        if os.path.exists(attack_dir):
            for img_name in os.listdir(attack_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(attack_dir, img_name))
                    self.labels.append(0.0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        # Etiketi BCE Loss'un istediği Float Tensor formatına çeviriyoruz
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

# ==========================================
# 2. MİMARİ: RESNET-18 (BCE Uyumlu)
# ==========================================
def get_resnet18_bce(device):
    # PSD & ADD Belgeleri: ResNet-18 kullanılacak
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # ADD Belgesi: Binary Cross Entropy (BCE) Loss kullanılacak.
    # BCE Loss tek bir çıktı düğümü (node) bekler (Sadece 1 olma olasılığı)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1) 
    
    return model.to(device)

# ==========================================
# 3. METRİK HESAPLAMA (HTER, AUC-ROC vb.)
# ==========================================
def calculate_fpad_metrics(y_true, y_prob):
    # Olasılıkları 0.5 eşik değeriyle 1 veya 0 sınıfına ayırıyoruz
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]

    TP = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    TN = sum((yt == 0 and yp == 0) for yt, yp in zip(y_true, y_pred))
    FP = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    FN = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))

    apcer = FP / (FP + TN + 1e-8)
    bpcer = FN / (FN + TP + 1e-8)
    
    # ADD Belgesinde istenen HTER (Half Total Error Rate), ACER ile birebir aynıdır.
    hter = (apcer + bpcer) / 2.0
    
    # ADD Belgesinde istenen AUC-ROC skoru
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0 # Eğer sadece tek sınıf varsa hata vermemesi için

    return apcer, bpcer, hter, auc

# ==========================================
# 4. ANA EĞİTİM DÖNGÜSÜ
# ==========================================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Eğitim başlıyor... Kullanılan Donanım: {device}")

    # --- DOSYA YOLLARI ---
    train_dir = "/home/undergrad25_1/Desktop/fpad_diffusion/data/processed/ReplayAttack/train"
    dev_dir = "/home/undergrad25_1/Desktop/fpad_diffusion/data/processed/ReplayAttack/dev"

    # ADD Belgesi Bölüm 5.1: 256x256 standart çözünürlük
    fpad_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Veri setleri yükleniyor...")
    train_dataset = ReplayAttackDataset(root_dir=train_dir, transform=fpad_transform)
    dev_dataset = ReplayAttackDataset(root_dir=dev_dir, transform=fpad_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Mimariyi ve BCE Loss'u Başlat
    model = get_resnet18_bce(device)
    # BCEWithLogitsLoss, Sigmoid aktivasyonu ile BCE Loss'u birleştirir (Matematiksel olarak daha stabildir)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 20
    best_hter = float('inf')

    for epoch in range(num_epochs):
        # --- TRAIN FAZI ---
        model.train()
        running_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Eğitim]", leave=False, mininterval=10.0, ascii=True)
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            # Çıktıyı [Batch, 1] den [Batch] boyutuna indirgiyoruz ki label ile eşleşsin
            outputs = model(inputs).squeeze(1) 
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # --- DEV (VALIDATION) FAZI ---
        model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in dev_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze(1)
                
                # Çıktıları Sigmoid ile 0-1 arası olasılıklara (Probability) çeviriyoruz
                probs = torch.sigmoid(outputs)
                
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Metrikleri Hesapla
        apcer, bpcer, hter, auc = calculate_fpad_metrics(all_labels, all_probs)
        print(f"\nEpoch {epoch+1}/{num_epochs} Tamamlandı. | Train Loss (BCE): {running_loss/len(train_loader):.4f}")
        print(f"DEV Sonuçları -> APCER: {apcer*100:.2f}% | BPCER: {bpcer*100:.2f}% | HTER(ACER): {hter*100:.2f}% | AUC-ROC: {auc:.4f}")

        # En iyi modeli kaydet (HTER düştükçe)
        if hter < best_hter:
            best_hter = hter
            torch.save(model.state_dict(), "resnet18_baseline_best.pth")
            print(f"*** Yeni en iyi model kaydedildi! (HTER: {hter*100:.2f}%) ***")
        print("-" * 60)

if __name__ == "__main__":
    train_model()
