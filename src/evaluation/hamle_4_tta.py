"""
FPAD - HAMLE 4: Test Time Augmentation (TTA)
==============================================
Eğitime dokunmadan, sadece test sırasında her görüntüyü birden fazla
versiyonda inference geçirip tahminleri ortalama alır.

5 versiyonu birleştirir:
  1. Orijinal
  2. Yatay flip
  3. Hafif parlaklık artışı
  4. Hafif parlaklık azalışı
  5. Center crop (90% area)

Tüm yepyeni_resnet/ klasöründeki modeller için intra ve cross test uygular.

ÇIKTI:
  - tta_intra_results.txt
  - tta_cross_results.txt
  - tta_comparison_summary.txt   ← TEZ İÇİN

ÇALIŞTIRMA:
    python hamle_4_tta.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from tqdm import tqdm


# ==========================================
# VERİ OKUMA
# ==========================================
class FlexibleDataset(Dataset):
    """5 TTA varyantını birlikte döndürür"""
    def __init__(self, root_dir, tta_transforms):
        self.image_paths = []
        self.labels = []
        self.tta_transforms = tta_transforms
        valid_ext = ('.png', '.jpg', '.jpeg')

        real_base = os.path.join(root_dir, 'real')
        if os.path.exists(real_base):
            for root, _, files in os.walk(real_base):
                for f in files:
                    if f.lower().endswith(valid_ext):
                        self.image_paths.append(os.path.join(root, f))
                        self.labels.append(1.0)

        attack_base = os.path.join(root_dir, 'attack')
        if os.path.exists(attack_base):
            for root, _, files in os.walk(attack_base):
                for f in files:
                    if f.lower().endswith(valid_ext):
                        self.image_paths.append(os.path.join(root, f))
                        self.labels.append(0.0)

        print(f"  [{os.path.basename(root_dir)}] Real: {self.labels.count(1.0)}, "
              f"Attack: {self.labels.count(0.0)}, Toplam: {len(self.labels)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            # 5 TTA varyantı üret
            variants = [t(img) for t in self.tta_transforms]
            return torch.stack(variants), torch.tensor(self.labels[idx], dtype=torch.float32)
        except Exception:
            return torch.zeros(5, 3, 256, 256), torch.tensor(self.labels[idx], dtype=torch.float32)


# ==========================================
# TTA TRANSFORMS - 5 farklı görüntü versiyonu
# ==========================================
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

tta_transforms = [
    # 1. Orijinal
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ]),
    # 2. Yatay flip
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ]),
    # 3. Hafif parlaklık artışı
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ColorJitter(brightness=(1.1, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ]),
    # 4. Hafif parlaklık azalışı
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ColorJitter(brightness=(0.9, 0.9)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ]),
    # 5. Center crop (90%)
    transforms.Compose([
        transforms.Resize((284, 284)),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ]),
]


# ==========================================
# MODEL YÜKLEME
# ==========================================
def load_model(model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features

    if 'fc.1.weight' in state_dict:
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 1)
        )
    else:
        model.fc = nn.Linear(in_features, 1)

    model.load_state_dict(state_dict)
    model.to(device)
    return model


# ==========================================
# TTA INFERENCE
# ==========================================
def collect_predictions_tta(model, loader, device):
    """5 TTA varyantın tahminlerini ortalama alır"""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="  TTA Inference", leave=False):
            # inputs: [B, 5, 3, 256, 256]
            B, N, C, H, W = inputs.shape
            inputs = inputs.view(B*N, C, H, W).to(device)
            outputs = model(inputs).squeeze(1)
            probs = torch.sigmoid(outputs)
            # 5 versiyonun ortalaması
            probs = probs.view(B, N).mean(dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)


# ==========================================
# METRİKLER
# ==========================================
def compute_metrics_at_threshold(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    apcer = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    bpcer = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    hter = (apcer + bpcer) / 2.0
    return apcer * 100, bpcer * 100, hter * 100


def find_eer_threshold(probs, labels):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    return float(thresholds[eer_idx]), float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)


def evaluate(probs, labels):
    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0
    eer_thr, eer_val = find_eer_threshold(probs, labels)
    apcer_05, bpcer_05, hter_05 = compute_metrics_at_threshold(probs, labels, 0.5)
    apcer_eer, bpcer_eer, hter_eer = compute_metrics_at_threshold(probs, labels, eer_thr)
    return {
        'AUC': round(auc, 4),
        'EER (%)': round(eer_val * 100, 4),
        'HTER@0.5 (%)': round(hter_05, 4),
        'HTER@EER (%)': round(hter_eer, 4),
        'APCER@0.5 (%)': round(apcer_05, 4),
        'BPCER@0.5 (%)': round(bpcer_05, 4),
    }


# ==========================================
# ANA SÜREÇ
# ==========================================
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*75}")
    print(f"HAMLE 4: TEST TIME AUGMENTATION (TTA) EVALUATION")
    print(f"Donanım: {device}")
    print(f"{'='*75}\n")
    print("TTA stratejisi: 5 varyant (orijinal + flip + 2x brightness + center crop)")
    print("Her görüntü 5 kez inference geçirilir, ortalaması alınır.\n")

    MODEL_DIR = os.path.expanduser("~/Desktop/fpad_diffusion/results/yepyeni_resnet")
    OUTPUT_DIR = os.path.expanduser("~/Desktop/fpad_diffusion/results")
    OULU_TEST = os.path.expanduser("~/Desktop/fpad_diffusion/data/processed/OULU/test")
    REPLAY_TEST = os.path.expanduser("~/Desktop/fpad_diffusion/data/processed/ReplayAttack/test")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Test veri setleri yükleniyor (TTA için)...")
    oulu_ds = FlexibleDataset(OULU_TEST, tta_transforms)
    replay_ds = FlexibleDataset(REPLAY_TEST, tta_transforms)
    # batch_size düşük çünkü her örnek 5 görüntü içeriyor
    oulu_loader = DataLoader(oulu_ds, batch_size=8, shuffle=False, num_workers=4)
    replay_loader = DataLoader(replay_ds, batch_size=8, shuffle=False, num_workers=4)

    pth_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')])
    print(f"\nTest edilecek model sayısı: {len(pth_files)}\n")

    all_results = []
    for model_name in pth_files:
        model_path = os.path.join(MODEL_DIR, model_name)
        print(f"\n{'─'*75}")
        print(f"Model: {model_name}")
        print(f"{'─'*75}")

        try:
            model = load_model(model_path, device)
        except Exception as e:
            print(f"  HATA: {e}")
            continue

        if "oulu" in model_name.lower():
            train_set = "OULU-NPU"
            intra_loader = oulu_loader
            cross_loader = replay_loader
            cross_set = "Replay-Attack"
        elif "replay" in model_name.lower():
            train_set = "Replay-Attack"
            intra_loader = replay_loader
            cross_loader = oulu_loader
            cross_set = "OULU-NPU"
        else:
            continue

        print(f"  [Intra+TTA] {train_set}")
        intra_probs, intra_labels = collect_predictions_tta(model, intra_loader, device)
        intra = evaluate(intra_probs, intra_labels)

        print(f"  [Cross+TTA] {train_set} → {cross_set}")
        cross_probs, cross_labels = collect_predictions_tta(model, cross_loader, device)
        cross = evaluate(cross_probs, cross_labels)

        print(f"\n  INTRA AUC: {intra['AUC']:.4f} | HTER@EER: %{intra['HTER@EER (%)']:.2f}")
        print(f"  CROSS AUC: {cross['AUC']:.4f} | HTER@EER: %{cross['HTER@EER (%)']:.2f}")

        all_results.append({
            'model': model_name,
            'train_set': train_set,
            'cross_set': cross_set,
            'intra': intra,
            'cross': cross,
        })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ÇIKTILAR
    write_results(all_results, OUTPUT_DIR)


def write_results(all_results, output_dir):
    summary_path = os.path.join(output_dir, "tta_comparison_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("FPAD - TTA (TEST TIME AUGMENTATION) SONUÇLARI\n")
        f.write("=" * 95 + "\n\n")
        f.write("TTA: Her test görüntüsü 5 varyantta (orijinal + flip + 2x brightness + crop)\n")
        f.write("inference geçirilir, tahminler ortalanır.\n\n")

        f.write(">>> CROSS-DATASET (TTA ile) <<<\n")
        f.write("-" * 95 + "\n")
        f.write(f"{'Model':<48} {'AUC':>7}  {'HTER@0.5':>10}  {'HTER@EER':>10}\n")
        f.write("-" * 95 + "\n")
        f.write("\n[OULU → Replay-Attack]\n")
        for r in all_results:
            if 'oulu' in r['model'].lower():
                res = r['cross']
                f.write(f"{r['model']:<48} {res['AUC']:>7.3f}  "
                        f"%{res['HTER@0.5 (%)']:>8.2f}  "
                        f"%{res['HTER@EER (%)']:>8.2f}\n")
        f.write("\n[Replay-Attack → OULU-NPU]\n")
        for r in all_results:
            if 'replay' in r['model'].lower():
                res = r['cross']
                f.write(f"{r['model']:<48} {res['AUC']:>7.3f}  "
                        f"%{res['HTER@0.5 (%)']:>8.2f}  "
                        f"%{res['HTER@EER (%)']:>8.2f}\n")

        f.write("\n\n>>> INTRA-DATASET (TTA ile) <<<\n")
        f.write("-" * 95 + "\n")
        f.write(f"{'Model':<48} {'AUC':>7}  {'HTER@0.5':>10}  {'HTER@EER':>10}\n")
        f.write("-" * 95 + "\n")
        for r in all_results:
            res = r['intra']
            f.write(f"{r['model']:<48} {res['AUC']:>7.3f}  "
                    f"%{res['HTER@0.5 (%)']:>8.2f}  "
                    f"%{res['HTER@EER (%)']:>8.2f}\n")

        f.write("\n" + "=" * 95 + "\n")
        f.write("KARŞILAŞTIRMA İÇİN: TTA'sız sonuçları yepyeni_comparison_summary.txt'den görün\n")
        f.write("Beklenti: TTA, cross-dataset AUC'sini %1-3 artırmalı.\n")

    print(f"\n{'='*75}")
    print(f"✅ TTA SONUÇLARI KAYDEDİLDİ")
    print(f"{'='*75}")
    print(f"  {summary_path}")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
