"""
OULU SINIFLANDIRICILARI - TEMİZ DDPM/S2S İLE
=============================================
5 yeni OULU-NPU sınıflandırıcısı tek scriptte eğitir. Titan V (GPU 0) kullanılır.

NOT: Bu kod, train_clean_replay_classifiers.py'nin OULU'ya uyarlanmış halidir.
     Replay versiyonundan farkları:
       - GPU 1 (GTX 1080) yerine GPU 0 (Titan V)
       - Tüm yollar OULU-NPU klasör yapısına göre
       - Model dosya adlarında 'oulu' geçer (eval scripti 'oulu' in model_name
         kontrolü yaptığı için ZORUNLU — intra=OULU, cross=Replay ataması buna bağlı)

KONFİGÜRASYONLAR (Replay ile birebir aynı doz çalışması):
  1. OULU + 500 temiz DDPM
  2. OULU + 2K temiz DDPM
  3. OULU + 5K temiz DDPM
  4. OULU + 10K temiz DDPM
  5. OULU + 2K temiz DDPM + 500 temiz S2S   *EN KRİTİK KARŞILAŞTIRMA*

ÇIKTI:
  Modeller: ~/Desktop/fpad_diffusion/results/yepyeni_resnet/
    - resnet18_oulu_clean_ddpm_500.pth
    - resnet18_oulu_clean_ddpm_2000.pth
    - resnet18_oulu_clean_ddpm_5000.pth
    - resnet18_oulu_clean_ddpm_10000.pth
    - resnet18_oulu_clean_ddpm_s2s.pth

  Log: ~/Desktop/fpad_diffusion/results/yepyeni_resnet/clean_oulu_history.txt

ÇALIŞTIRMA:
    nohup python train_clean_oulu_classifiers.py > train_clean_oulu.log 2>&1 &

SONRAKİ ADIM (training bittikten sonra):
    python eer_threshold_eval.py
    # Yeni OULU modelleri otomatik test edilir (model adında 'oulu' geçtiği için
    # intra=OULU-NPU, cross=Replay-Attack olarak doğru atanır)
"""

import os

# ==========================================
# GPU SEÇİMİ - Titan V (GPU 0)
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve


# ==========================================
# YOLLAR — OULU-NPU
# ==========================================
USER_HOME = "/home/undergrad25_1"
DATA_ROOT = f"{USER_HOME}/Desktop/fpad_diffusion/data"

# Orijinal OULU (real/attack alt klasörleri içerir — Replay ile aynı yapı,
# sadece kök klasör adı 'OULU')
OULU_TRAIN = f"{DATA_ROOT}/processed/OULU/train"
OULU_DEV = f"{DATA_ROOT}/processed/OULU/dev"

# Temiz DDPM sentetik (OULU) — senin verdiğin yol
CLEAN_DDPM_OULU = f"{DATA_ROOT}/synthetic/OULU-NPU/DDPM_spoof_fixed/spoof"

# Temiz S2S sentetik (OULU) — senin verdiğin yol
CLEAN_S2S_OULU = f"{DATA_ROOT}/S2S_fixed/OULU-NPU"

# Çıktı (Replay ile aynı dizin, eval scripti buradan okuyor)
OUTPUT_DIR = f"{USER_HOME}/Desktop/fpad_diffusion/results/yepyeni_resnet"


# ==========================================
# VERİ SINIFLARI
# ==========================================
class FlexibleDataset(Dataset):
    """Real ve attack klasörlerini birlikte okur.
    OULU iç içe klasör yapısına sahip olabileceği için os.walk ile recursive tarar."""
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        valid_ext = ('.png', '.jpg', '.jpeg')

        real_dir = os.path.join(root_dir, 'real')
        if os.path.exists(real_dir):
            for subdir, _, files in os.walk(real_dir):
                for f in files:
                    if f.lower().endswith(valid_ext):
                        self.image_paths.append(os.path.join(subdir, f))
                        self.labels.append(1.0)

        attack_dir = os.path.join(root_dir, 'attack')
        if os.path.exists(attack_dir):
            for subdir, _, files in os.walk(attack_dir):
                for f in files:
                    if f.lower().endswith(valid_ext):
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


class FlatFolderDataset(Dataset):
    """Sentetik verilerin düz klasörü için"""
    def __init__(self, root_dir, label, transform=None, max_samples=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        valid_ext = ('.png', '.jpg', '.jpeg')

        if not os.path.exists(root_dir):
            print(f"  UYARI: Klasör bulunamadı: {root_dir}")
            return

        # Düz klasör
        for f in os.listdir(root_dir):
            full_path = os.path.join(root_dir, f)
            if os.path.isfile(full_path) and f.lower().endswith(valid_ext):
                self.image_paths.append(full_path)
                self.labels.append(float(label))

        # Eğer düz klasörde yoksa recursive ara
        if len(self.image_paths) == 0:
            for subdir, _, files in os.walk(root_dir):
                for f in files:
                    if f.lower().endswith(valid_ext):
                        self.image_paths.append(os.path.join(subdir, f))
                        self.labels.append(float(label))

        if max_samples and len(self.image_paths) > max_samples:
            random.seed(42)
            idx = random.sample(range(len(self.image_paths)), max_samples)
            self.image_paths = [self.image_paths[i] for i in idx]
            self.labels = [self.labels[i] for i in idx]

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
# AUGMENTATION
# ==========================================
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ==========================================
# MODEL
# ==========================================
def get_resnet18_with_dropout(device, dropout_p=0.3):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features, 1)
    )
    return model.to(device)


# ==========================================
# METRİKLER
# ==========================================
def compute_metrics(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    hter_eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    y_pred = (y_prob >= 0.5).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    apcer = fp / (fp + tn + 1e-8)
    bpcer = fn / (fn + tp + 1e-8)
    hter_05 = (apcer + bpcer) / 2.0
    return {'auc': auc, 'hter_05': hter_05 * 100, 'hter_eer': hter_eer * 100,
            'apcer_05': apcer * 100, 'bpcer_05': bpcer * 100}


# ==========================================
# EĞİTİM FONKSİYONU
# ==========================================
def train_config(name, synth_sources, save_path):
    print(f"\n{'='*75}")
    print(f"KONFİGÜRASYON: {name}")
    print(f"{'='*75}")

    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # CUDA_VISIBLE_DEVICES=0 olduğu için cuda:0 → Titan V

    # Orijinal veri
    orig_dataset = FlexibleDataset(OULU_TRAIN, transform=train_transform)
    real_count = orig_dataset.labels.count(1.0)
    attack_count = orig_dataset.labels.count(0.0)
    print(f"  Orijinal OULU: Real={real_count}, Attack={attack_count}")

    # Sentetik kaynakları ekle
    datasets_to_concat = [orig_dataset]
    total_synth = 0
    synth_summary = []
    for synth_path, max_samples in synth_sources:
        synth_ds = FlatFolderDataset(synth_path, label=0.0,
                                      transform=train_transform,
                                      max_samples=max_samples)
        if len(synth_ds) > 0:
            datasets_to_concat.append(synth_ds)
            total_synth += len(synth_ds)
            synth_summary.append(f"{os.path.basename(synth_path)}({len(synth_ds)})")
            print(f"  Sentetik [{os.path.basename(synth_path)}]: {len(synth_ds)}")
        else:
            print(f"  UYARI: {synth_path} - boş veya bulunamadı")

    train_ds = ConcatDataset(datasets_to_concat)
    total_attack = attack_count + total_synth
    print(f"\n  >>> TOPLAM EĞİTİM SETİ <<<")
    print(f"  Real:    {real_count}")
    print(f"  Attack:  {total_attack} (orig {attack_count} + sentetik {total_synth})")
    print(f"  TOPLAM:  {len(train_ds)}")
    if real_count > 0:
        print(f"  Attack/Real oranı: {total_attack/real_count:.2f}:1\n")
    else:
        print(f"  UYARI: Real sayısı 0! OULU_TRAIN yolunu kontrol et: {OULU_TRAIN}\n")

    dev_ds = FlexibleDataset(OULU_DEV, transform=test_transform)
    print(f"  Dev: {len(dev_ds)}\n")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, num_workers=4)

    model = get_resnet18_with_dropout(device, dropout_p=0.3)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    EPOCHS = 8
    PATIENCE = 3
    best_auc = 0.0
    no_improve = 0
    history = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader,
                                    desc=f"  Epoch {epoch+1}/{EPOCHS}",
                                    leave=False, mininterval=15):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)

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
            torch.save(model.state_dict(), save_path)
            print(f"    >> YENİ EN İYİ AUC, kaydedildi: {os.path.basename(save_path)}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping (patience={PATIENCE})")
                break

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'name': name,
        'save_path': save_path,
        'best_auc': best_auc,
        'history': history,
        'total_samples': len(train_ds),
        'real_count': real_count,
        'attack_count': total_attack,
        'synth_summary': ', '.join(synth_summary) if synth_summary else 'yok',
    }


# ==========================================
# ANA SÜREÇ
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'#'*75}")
    print(f"# TEMİZ OULU-NPU SINIFLANDIRICILARI")
    print(f"# GPU: Titan V (CUDA_VISIBLE_DEVICES=0)")
    print(f"# Çıktı dizini: {OUTPUT_DIR}")
    print(f"{'#'*75}")

    # ==========================================
    # 5 KONFİGÜRASYON (Replay ile birebir aynı doz çalışması)
    # ==========================================
    configs = [
        {
            'name': '[1/5] OULU - Real + 500 TEMİZ DDPM',
            'synth_sources': [(CLEAN_DDPM_OULU, 500)],
            'save_path': os.path.join(OUTPUT_DIR, 'resnet18_oulu_clean_ddpm_500.pth'),
        },
        {
            'name': '[2/5] OULU - Real + 2K TEMİZ DDPM',
            'synth_sources': [(CLEAN_DDPM_OULU, 2000)],
            'save_path': os.path.join(OUTPUT_DIR, 'resnet18_oulu_clean_ddpm_2000.pth'),
        },
        {
            'name': '[3/5] OULU - Real + 5K TEMİZ DDPM',
            'synth_sources': [(CLEAN_DDPM_OULU, 5000)],
            'save_path': os.path.join(OUTPUT_DIR, 'resnet18_oulu_clean_ddpm_5000.pth'),
        },
        {
            'name': '[4/5] OULU - Real + 10K TEMİZ DDPM',
            'synth_sources': [(CLEAN_DDPM_OULU, 10000)],
            'save_path': os.path.join(OUTPUT_DIR, 'resnet18_oulu_clean_ddpm_10000.pth'),
        },
        {
            'name': '[5/5] OULU - Real + 2K TEMİZ DDPM + 500 TEMİZ S2S **EN KRİTİK**',
            'synth_sources': [
                (CLEAN_DDPM_OULU, 2000),
                (CLEAN_S2S_OULU, 500),
            ],
            'save_path': os.path.join(OUTPUT_DIR, 'resnet18_oulu_clean_ddpm_s2s.pth'),
        },
    ]

    results = []
    for i, cfg in enumerate(configs, 1):
        print(f"\n\n{'#'*75}")
        print(f"# DENEY {i}/{len(configs)}")
        print(f"{'#'*75}")
        try:
            res = train_config(
                cfg['name'],
                cfg['synth_sources'],
                cfg['save_path'],
            )
            results.append(res)
        except Exception as e:
            print(f"\n  HATA: {e}")
            results.append({'name': cfg['name'], 'error': str(e)})

    # ÖZET
    print(f"\n\n{'='*75}")
    print(f"TEMİZ OULU SONUÇ ÖZETİ")
    print(f"{'='*75}\n")

    log_path = os.path.join(OUTPUT_DIR, 'clean_oulu_history.txt')
    with open(log_path, 'w') as f:
        f.write("FPAD - TEMİZ OULU-NPU SINIFLANDIRICILARI\n")
        f.write("=" * 75 + "\n\n")

        f.write(f"{'Konfigürasyon':<55} {'Toplam':>8} {'DEV AUC':>10}\n")
        f.write("-" * 75 + "\n")
        for r in results:
            if 'error' in r:
                f.write(f"{r['name']:<55} {'HATA':>8} {'-':>10}\n")
                print(f"❌ {r['name']}: HATA - {r['error']}")
            else:
                f.write(f"{r['name']:<55} {r['total_samples']:>8} "
                        f"{r['best_auc']:>10.4f}\n")
                print(f"✅ {r['name']}")
                print(f"   Toplam={r['total_samples']}, Best AUC={r['best_auc']:.4f}")

        f.write("\n\nDETAYLI EPOCH GEÇMİŞİ\n")
        f.write("=" * 75 + "\n\n")
        for r in results:
            if 'error' in r:
                continue
            f.write(f"\n{r['name']}\n")
            f.write(f"  Model: {r['save_path']}\n")
            f.write(f"  Sentetik: {r['synth_summary']}\n")
            for h in r['history']:
                f.write(f"    Epoch {h['epoch']}: loss={h['loss']:.4f}, "
                        f"AUC={h['auc']:.4f}, HTER@EER=%{h['hter_eer']:.2f}\n")
            f.write("-" * 75 + "\n")

    print(f"\n{'='*75}")
    print(f"✅ TAMAMLANDI")
    print(f"{'='*75}")
    print(f"  Yeni modeller: {OUTPUT_DIR}/")
    print(f"  Log:           {log_path}")
    print(f"\nSONRAKİ ADIM:")
    print(f"  python eer_threshold_eval.py")
    print(f"  (OULU modelleri 'oulu' içerdiği için intra=OULU, cross=Replay olarak test edilir)")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
