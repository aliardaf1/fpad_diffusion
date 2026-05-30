"""
FPAD - OULU Subsample Training (Strateji 3)
=============================================
- Orijinal eğitim oranı korunur (Attack:Real = 4:1)
- Sentetik attack ile orijinal attack EŞIT (16K + 16K)
- Color-robust augmentation: DDPM renk cast'ine bağımlılığı kırar
- Weighted Sampler yerine RANDOM subsample (daha temiz metodoloji)

Çalıştırma:
    python train_oulu_subsample.py
"""

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np


# ==========================================
# 1. VERİ OKUMA SINIFLARI
# ==========================================
class OuluDataset(Dataset):
    """OULU iç içe klasör yapısı (os.walk ile derin okuma)"""
    def __init__(self, root_dir, transform=None, only_label=None):
        """
        only_label: None=hepsi, 0.0=sadece attack, 1.0=sadece real
        """
        self.image_paths = []
        self.labels = []
        self.transform = transform

        real_dir = os.path.join(root_dir, 'real')
        if os.path.exists(real_dir) and (only_label is None or only_label == 1.0):
            for subdir, _, files in os.walk(real_dir):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(subdir, f))
                        self.labels.append(1.0)

        attack_dir = os.path.join(root_dir, 'attack')
        if os.path.exists(attack_dir) and (only_label is None or only_label == 0.0):
            for subdir, _, files in os.walk(attack_dir):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(subdir, f))
                        self.labels.append(0.0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(self.labels[idx], dtype=torch.float32)
        except Exception:
            return torch.zeros(3, 256, 256), torch.tensor(self.labels[idx], dtype=torch.float32)


class SimpleFolderDataset(Dataset):
    """Düz klasör (sentetik veri için)"""
    def __init__(self, root_dir, label, transform=None):
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.labels = [float(label)] * len(self.image_paths)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(self.labels[idx], dtype=torch.float32)
        except Exception:
            return torch.zeros(3, 256, 256), torch.tensor(self.labels[idx], dtype=torch.float32)


# ==========================================
# 2. AUGMENTATION — DDPM artifact'lerine karşı zırh
# ==========================================
# Eğitim transform'u — agresif color augmentation
# Mantık: DDPM görüntülerimizde belirgin renk cast'leri var (yeşil/mavi/sarı/kırmızı).
# Eğer model bu renk cast'ini "spoof imzası" olarak öğrenirse, cross-dataset'te
# farklı renk cast'leri olan Replay-Attack'ı tanıyamaz. ColorJitter ile her epoch'ta
# rastgele renk değişiklikleri ekleyerek modeli "rengi ignore etmeye" zorluyoruz.
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),

    # KRİTİK: Renk cast'ine bağımlılığı kır
    transforms.ColorJitter(
        brightness=0.3,    # Parlaklık ±30%
        contrast=0.3,      # Kontrast ±30%
        saturation=0.5,    # Doygunluk ±50% (renk yoğunluğu)
        hue=0.15           # Renk tonu ±15% — en kritik parametre
    ),
    transforms.RandomGrayscale(p=0.1),  # %10 ihtimalle gri tonlama

    transforms.ToTensor(),

    # Hafif gauss noise — sentetik ile gerçek arasındaki "temizlik farkını" kapat
    transforms.RandomApply([
        transforms.Lambda(lambda x: torch.clamp(x + 0.03 * torch.randn_like(x), 0, 1))
    ], p=0.3),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Test transform'u — augmentation YOK, sadece normalize
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ==========================================
# 3. MODEL & METRİKLER
# ==========================================
def get_resnet18(device):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model.to(device)


def compute_metrics(y_true, y_prob):
    """HTER, AUC, EER threshold hesaplar."""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # AUC
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0

    # EER threshold (APCER == BPCER noktası)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer_thr = thresholds[eer_idx]
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0

    # 0.5 threshold ile HTER
    y_pred = (y_prob >= 0.5).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    apcer = fp / (fp + tn + 1e-8)
    bpcer = fn / (fn + tp + 1e-8)
    hter = (apcer + bpcer) / 2.0

    # EER threshold ile HTER
    y_pred_eer = (y_prob >= eer_thr).astype(int)
    tp_e = ((y_pred_eer == 1) & (y_true == 1)).sum()
    tn_e = ((y_pred_eer == 0) & (y_true == 0)).sum()
    fp_e = ((y_pred_eer == 1) & (y_true == 0)).sum()
    fn_e = ((y_pred_eer == 0) & (y_true == 1)).sum()
    apcer_e = fp_e / (fp_e + tn_e + 1e-8)
    bpcer_e = fn_e / (fn_e + tp_e + 1e-8)
    hter_eer = (apcer_e + bpcer_e) / 2.0

    return {
        'auc': auc,
        'hter_05': hter * 100,
        'hter_eer': hter_eer * 100,
        'apcer_05': apcer * 100,
        'bpcer_05': bpcer * 100,
        'eer_threshold': eer_thr,
    }


# ==========================================
# 4. EĞİTİM
# ==========================================
def train():
    # Tekrarlanabilirlik için seed
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"OULU SUBSAMPLE TRAINING (Strateji 3)")
    print(f"Cihaz: {device}")
    print(f"{'='*70}\n")

    # --- YOLLAR ---
    ORIG_TRAIN = "/home/undergrad25_1/Desktop/fpad_diffusion/data/processed/OULU/train"
    DEV_ROOT = "/home/undergrad25_1/Desktop/fpad_diffusion/data/processed/OULU/dev"
    SYNTH_ATTACK = "/home/undergrad25_1/Desktop/fpad_diffusion/data/synthetic/OULU-NPU/DDPM_spoof/spoof"

    # ==========================================
    # SUBSAMPLE STRATEJİSİ
    # ==========================================
    # OULU train: 8003 real + 32538 attack (orig oran 4:1)
    # Hedef: 8003 real + 16000 orig attack + 16000 sentetik attack = 40003 toplam
    # Attack:Real = 4:1 (baseline ile AYNI oran, ADİL karşılaştırma)
    # Sentetik:Orig attack oranı = 1:1 (DDPM destekleyici, domine etmiyor)

    TARGET_REAL = 8003
    TARGET_ORIG_ATTACK = 16000
    TARGET_SYNTH_ATTACK = 16000

    print("Veri setleri hazırlanıyor...")

    # 1) Orijinal Real (tamamı)
    orig_real = OuluDataset(ORIG_TRAIN, transform=train_transform, only_label=1.0)
    print(f"  Orijinal Real bulundu: {len(orig_real)}")

    # 2) Orijinal Attack — 16000'e subsample
    orig_attack_full = OuluDataset(ORIG_TRAIN, transform=train_transform, only_label=0.0)
    print(f"  Orijinal Attack bulundu: {len(orig_attack_full)}")
    attack_indices = random.sample(range(len(orig_attack_full)), TARGET_ORIG_ATTACK)
    orig_attack = Subset(orig_attack_full, attack_indices)

    # 3) Sentetik Attack — 16000'e subsample
    synth_full = SimpleFolderDataset(SYNTH_ATTACK, label=0.0, transform=train_transform)
    print(f"  Sentetik Attack bulundu: {len(synth_full)}")
    synth_indices = random.sample(range(len(synth_full)), TARGET_SYNTH_ATTACK)
    synth_attack = Subset(synth_full, synth_indices)

    # 4) Birleştir
    train_ds = ConcatDataset([orig_real, orig_attack, synth_attack])
    total_attack = len(orig_attack) + len(synth_attack)
    print(f"\n>>> EĞİTİM SETİ DAĞILIMI <<<")
    print(f"  Real:            {len(orig_real):>6} ({len(orig_real)/len(train_ds)*100:.1f}%)")
    print(f"  Orig Attack:     {len(orig_attack):>6} ({len(orig_attack)/len(train_ds)*100:.1f}%)")
    print(f"  Sentetik Attack: {len(synth_attack):>6} ({len(synth_attack)/len(train_ds)*100:.1f}%)")
    print(f"  TOPLAM:          {len(train_ds):>6}")
    print(f"  Attack:Real    = {total_attack/len(orig_real):.2f}:1  (baseline ile AYNI)")
    print(f"  Synth:Orig     = {len(synth_attack)/len(orig_attack):.2f}:1  (eşit karışım)\n")

    # 5) Dev seti (orijinal, augmentation yok)
    dev_ds = OuluDataset(DEV_ROOT, transform=test_transform)
    print(f"  Dev seti: {len(dev_ds)}")

    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, num_workers=4)

    # ==========================================
    # MODEL
    # ==========================================
    model = get_resnet18(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # Learning rate scheduler — son epoch'larda LR düşür
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    # ==========================================
    # EĞİTİM DÖNGÜSÜ
    # ==========================================
    EPOCHS = 20
    best_dev_auc = 0.0
    OUTPUT_DIR = os.path.expanduser("~/Desktop/fpad_diffusion/results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_save_path = os.path.join(OUTPUT_DIR, "resnet18_oulu_subsample_best.pth")

    history = []

    print(f"\n{'='*70}")
    print(f"EĞİTİM BAŞLIYOR — {EPOCHS} epoch")
    print(f"{'='*70}\n")

    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", mininterval=10)

        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        scheduler.step()

        # --- VALIDATION ---
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in dev_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze(1)
                all_probs.extend(torch.sigmoid(outputs).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        m = compute_metrics(all_labels, all_probs)

        print(f"\nEpoch {epoch+1:2d} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"  DEV → AUC: {m['auc']:.4f} | HTER@0.5: %{m['hter_05']:.2f} | "
              f"HTER@EER: %{m['hter_eer']:.2f}")
        print(f"  DEV → APCER: %{m['apcer_05']:.2f} | BPCER: %{m['bpcer_05']:.2f}")

        history.append({'epoch': epoch+1, 'loss': avg_loss, **m})

        # En iyi AUC'yi kaydet (HTER değil, çünkü AUC threshold'dan bağımsız)
        if m['auc'] > best_dev_auc:
            best_dev_auc = m['auc']
            torch.save(model.state_dict(), best_save_path)
            print(f"  >> YENİ EN İYİ AUC ({m['auc']:.4f}) — kaydedildi: {best_save_path}")

        print("-" * 70)

    # ==========================================
    # ÖZET
    # ==========================================
    print(f"\n{'='*70}")
    print(f"EĞİTİM TAMAMLANDI")
    print(f"{'='*70}")
    print(f"En iyi DEV AUC: {best_dev_auc:.4f}")
    print(f"Model dosyası:  {best_save_path}")
    print(f"\nKarşılaştırma için:")
    print(f"  oulu_baseline   AUC = 0.668 | HTER@EER = %33.97")
    print(f"  oulu_augmented  AUC = 0.545 | HTER@EER = %44.95  (eski DDPM)")
    print(f"  oulu_subsample  AUC = {best_dev_auc:.3f} | HTER@EER = ?  (YENİ)")
    print(f"\nŞimdi cross-dataset test çalıştırın:")
    print(f"  python cross_dataset_eval.py")
    print(f"{'='*70}\n")

    # History'yi log dosyasına yaz
    history_path = os.path.join(OUTPUT_DIR, "oulu_subsample_history.txt")
    with open(history_path, 'w') as f:
        f.write("Epoch | Loss   | AUC    | HTER@0.5 | HTER@EER | APCER  | BPCER\n")
        f.write("-" * 70 + "\n")
        for h in history:
            f.write(f"{h['epoch']:5d} | {h['loss']:.4f} | {h['auc']:.4f} | "
                    f"%{h['hter_05']:>6.2f} | %{h['hter_eer']:>6.2f} | "
                    f"%{h['apcer_05']:>5.2f} | %{h['bpcer_05']:>5.2f}\n")
    print(f"Eğitim geçmişi kaydedildi: {history_path}\n")


if __name__ == "__main__":
    train()
