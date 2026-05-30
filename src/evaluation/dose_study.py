"""
FPAD - Sentetik Veri Doz Çalışması (Dose-Response Study)
==========================================================
Hipotez: Sentetik veri miktarı vs cross-dataset performansı

Denenen dozlar:
    OULU:   2K ve 8K sentetik attack
    Replay: 500 ve 2K sentetik attack

Toplam: 4 training (~12-15 saat)

Önceki bulgular ışığında değişiklikler:
- ColorJitter YUMUŞATILDI (hue=0.05, brightness=0.15) — önceki agresif aug intra'yı bozmuştu
- RandomGrayscale KALDIRILDI — gereksiz çıktı
- EPOCH 8'e indirildi — önceki training'lerde en iyi AUC epoch 1-3'te geliyordu
- Early stopping eklendi (patience=3)
- Tüm konfigürasyonlar TEK script'te — bash döngüsü yok, tek Python süreci

Çalıştırma:
    python dose_study.py
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve


# ==========================================
# 1. VERİ OKUMA SINIFLARI
# ==========================================
class FlexibleDataset(Dataset):
    """Hem OULU iç içe hem Replay düz klasör için çalışır"""
    def __init__(self, root_dir, transform=None, only_label=None):
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
    """Sentetik düz klasör"""
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
# 2. AUGMENTATION - YUMUŞATILMIŞ versiyon
# ==========================================
# Önceki agresif aug (hue=0.15, grayscale=0.1) intra-AUC'yi 0.999'dan 0.981'e düşürmüştü
# Bu sefer daha hafif: model renk ipuçlarını kullanabilsin ama ezberleyemesin
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.15,    # 0.3 -> 0.15
        contrast=0.15,      # 0.3 -> 0.15
        saturation=0.2,     # 0.5 -> 0.2
        hue=0.05            # 0.15 -> 0.05 (KRİTİK - çok yumuşadı)
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ==========================================
# 3. METRİKLER
# ==========================================
def compute_metrics(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer_thr = thresholds[eer_idx]
    hter_eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    return {'auc': auc, 'hter_eer': hter_eer * 100, 'eer_thr': float(eer_thr)}


# ==========================================
# 4. GENEL TRAINING FONKSİYONU
# ==========================================
def train_one_config(config):
    """
    Tek bir konfigürasyon için training yapar.
    config: dict — name, orig_train, dev_root, synth_dir, 
                   target_real, target_orig_attack, target_synth, save_path
    """
    print(f"\n{'='*70}")
    print(f"KONFİGÜRASYON: {config['name']}")
    print(f"{'='*70}")

    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- Veri setlerini hazırla ----
    orig_real = FlexibleDataset(config['orig_train'], transform=train_transform, only_label=1.0)
    orig_attack_full = FlexibleDataset(config['orig_train'], transform=train_transform, only_label=0.0)

    # Orig attack: full kullan (bu deneyde orig attack'i kısmıyoruz)
    print(f"  Orijinal Real:    {len(orig_real):>6}")
    print(f"  Orijinal Attack:  {len(orig_attack_full):>6}")

    # Sentetik attack: belirtilen miktarda subsample
    synth_full = SimpleFolderDataset(config['synth_dir'], label=0.0, transform=train_transform)
    print(f"  Sentetik havuzu:  {len(synth_full):>6}")
    
    if config['target_synth'] > 0:
        synth_indices = random.sample(range(len(synth_full)), 
                                       min(config['target_synth'], len(synth_full)))
        synth_attack = Subset(synth_full, synth_indices)
        train_ds = ConcatDataset([orig_real, orig_attack_full, synth_attack])
        synth_used = len(synth_attack)
    else:
        train_ds = ConcatDataset([orig_real, orig_attack_full])
        synth_used = 0

    total_attack = len(orig_attack_full) + synth_used
    print(f"\n  >>> EĞİTİM SETİ <<<")
    print(f"  Real:            {len(orig_real):>6}")
    print(f"  Orig Attack:     {len(orig_attack_full):>6}")
    print(f"  Sentetik Attack: {synth_used:>6}")
    print(f"  TOPLAM:          {len(train_ds):>6}")
    print(f"  Synth/Total Attack oranı: {synth_used/total_attack*100:.1f}%\n")

    dev_ds = FlexibleDataset(config['dev_root'], transform=test_transform)
    print(f"  Dev seti: {len(dev_ds)}\n")

    # ---- DataLoader ----
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, num_workers=4)

    # ---- Model ----
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ---- Training (8 epoch + early stopping) ----
    EPOCHS = 8
    PATIENCE = 3
    best_auc = 0.0
    no_improve = 0
    history = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"  Epoch {epoch+1}/{EPOCHS}", 
                                    leave=False, mininterval=15):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in dev_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze(1)
                all_probs.extend(torch.sigmoid(outputs).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        m = compute_metrics(all_labels, all_probs)
        print(f"  Epoch {epoch+1} | Loss: {avg_loss:.4f} | "
              f"DEV AUC: {m['auc']:.4f} | HTER@EER: %{m['hter_eer']:.2f}")

        history.append({'epoch': epoch+1, 'loss': avg_loss, **m})

        if m['auc'] > best_auc:
            best_auc = m['auc']
            torch.save(model.state_dict(), config['save_path'])
            print(f"    >> YENİ EN İYİ AUC, kaydedildi.")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping ({PATIENCE} epoch boyunca iyileşme yok)")
                break

    # GPU temizliği
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'name': config['name'],
        'best_auc': best_auc,
        'history': history,
        'synth_used': synth_used,
        'save_path': config['save_path'],
    }


# ==========================================
# 5. ANA SÜREÇ - 4 KONFİGÜRASYON
# ==========================================
def main():
    # ---- BASE PATHS ----
    OULU_TRAIN = "/home/undergrad25_1/Desktop/fpad_diffusion/data/processed/OULU/train"
    OULU_DEV = "/home/undergrad25_1/Desktop/fpad_diffusion/data/processed/OULU/dev"
    OULU_SYNTH = "/home/undergrad25_1/Desktop/fpad_diffusion/data/synthetic/OULU-NPU/DDPM_spoof/spoof"

    REPLAY_TRAIN = "/home/undergrad25_1/Desktop/fpad_diffusion/data/processed/ReplayAttack/train"
    REPLAY_DEV = "/home/undergrad25_1/Desktop/fpad_diffusion/data/processed/ReplayAttack/dev"
    REPLAY_SYNTH = "/home/undergrad25_1/Desktop/fpad_diffusion/data/synthetic/ReplayAttack/DDPM_spoof/spoof"

    # ---- 4 KONFİGÜRASYON ----
    configs = [
        {
            'name': 'OULU + 2K sentetik (mikro doz)',
            'orig_train': OULU_TRAIN,
            'dev_root': OULU_DEV,
            'synth_dir': OULU_SYNTH,
            'target_synth': 2000,
            'save_path': 'resnet18_oulu_dose2k_best.pth',
        },
        {
            'name': 'OULU + 8K sentetik (az doz)',
            'orig_train': OULU_TRAIN,
            'dev_root': OULU_DEV,
            'synth_dir': OULU_SYNTH,
            'target_synth': 8000,
            'save_path': 'resnet18_oulu_dose8k_best.pth',
        },
        {
            'name': 'Replay + 500 sentetik (mikro doz)',
            'orig_train': REPLAY_TRAIN,
            'dev_root': REPLAY_DEV,
            'synth_dir': REPLAY_SYNTH,
            'target_synth': 500,
            'save_path': 'resnet18_replay_dose500_best.pth',
        },
        {
            'name': 'Replay + 2K sentetik (az doz)',
            'orig_train': REPLAY_TRAIN,
            'dev_root': REPLAY_DEV,
            'synth_dir': REPLAY_SYNTH,
            'target_synth': 2000,
            'save_path': 'resnet18_replay_dose2k_best.pth',
        },
    ]

    # ---- HER KONFİGÜRASYONU SIRAYLA EĞIT ----
    results = []
    for i, cfg in enumerate(configs, 1):
        print(f"\n\n{'#'*70}")
        print(f"# DENEYE {i}/{len(configs)}: {cfg['name']}")
        print(f"{'#'*70}")
        try:
            result = train_one_config(cfg)
            results.append(result)
        except Exception as e:
            print(f"  HATA: {e}")
            results.append({'name': cfg['name'], 'best_auc': 0.0, 'error': str(e)})

    # ---- ÖZET RAPOR ----
    print(f"\n\n{'='*70}")
    print(f"DOZ ÇALIŞMASI - TOPLAM ÖZET")
    print(f"{'='*70}\n")

    # Karşılaştırma için baseline değerleri
    print(f"{'Konfigürasyon':<45} {'Sentetik':>10} {'Dev AUC':>10}")
    print("-" * 70)
    print(f"{'oulu_baseline (referans)':<45} {'0':>10} {'0.9986':>10}")
    print(f"{'oulu_augmented_eski (50K)':<45} {'50000':>10} {'0.9982':>10}")
    print(f"{'oulu_subsample (16K)':<45} {'16000':>10} {'0.9813':>10}")
    print("-" * 70)
    for r in results:
        if 'error' not in r:
            print(f"{r['name']:<45} {r['synth_used']:>10} {r['best_auc']:>10.4f}")
        else:
            print(f"{r['name']:<45} {'HATA':>10} {'-':>10}")

    print(f"\n{'='*70}")
    print(f"SIRADAKI ADIM: Tüm yeni modelleri Resnet18/ klasörüne kopyalayıp")
    print(f"               eer_threshold_eval.py çalıştırın")
    print(f"{'='*70}\n")

    # History dosyası
    with open('dose_study_history.txt', 'w') as f:
        f.write("FPAD - SENTETİK VERİ DOZ ÇALIŞMASI\n")
        f.write("="*70 + "\n\n")
        for r in results:
            f.write(f"Konfigürasyon: {r['name']}\n")
            if 'error' in r:
                f.write(f"  HATA: {r['error']}\n")
            else:
                f.write(f"  Sentetik kullanılan: {r['synth_used']}\n")
                f.write(f"  En iyi DEV AUC: {r['best_auc']:.4f}\n")
                f.write(f"  Model dosyası: {r['save_path']}\n")
                f.write(f"  Epoch geçmişi:\n")
                for h in r['history']:
                    f.write(f"    Epoch {h['epoch']}: loss={h['loss']:.4f}, "
                            f"AUC={h['auc']:.4f}, HTER@EER=%{h['hter_eer']:.2f}\n")
            f.write("-" * 70 + "\n")
    print("Detaylı geçmiş: dose_study_history.txt\n")


if __name__ == "__main__":
    main()
