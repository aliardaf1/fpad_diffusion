"""
FPAD - YEPYENİ RESNET MODELLERİ İÇİN EER THRESHOLD DEĞERLENDİRMESİ
====================================================================
yepyeni_resnet/ klasöründeki 8 yeni modeli test eder.

Çıktılar (results/ dizinine yazılır):
  - intra_dataset_yepyeni_results.txt
  - cross_dataset_yepyeni_results.txt
  - yepyeni_comparison_summary.txt  ← TEZ İÇİN ANA TABLO

"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from tqdm import tqdm


# ==========================================
# 1. VERİ OKUMA SINIFI
# ==========================================
class FlexibleFPADDataset(Dataset):
    """OULU iç içe klasörler ve Replay düz klasörler — ikisini de okur."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
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
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(self.labels[idx], dtype=torch.float32)
        except Exception:
            return torch.zeros(3, 256, 256), torch.tensor(self.labels[idx], dtype=torch.float32)


test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ==========================================
# 2. MODEL YÜKLEME — Dropout'lu FC için uyumlu
# ==========================================
def load_model(model_path, device):
    """
    Yeni modeller dropout'lu FC ile kaydedildi:
      model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(...))
    Bu fonksiyon hem dropout'lu hem de eski (sade Linear) yapıyı yükleyebilir.
    """
    state_dict = torch.load(model_path, map_location=device)

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features

    # State dict'i kontrol et — fc.1.weight varsa Sequential yapı (yeni)
    if 'fc.1.weight' in state_dict:
        # Dropout + Linear (yepyeni modellerde)
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 1)
        )
    else:
        # Sade Linear (eski modellerde)
        model.fc = nn.Linear(in_features, 1)

    model.load_state_dict(state_dict)
    model.to(device)
    return model


# ==========================================
# 3. ÇIKARIM
# ==========================================
def collect_predictions(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="  Inference", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze(1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)


# ==========================================
# 4. METRİK HESAPLAMA
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


def evaluate_with_both_thresholds(probs, labels):
    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0
    eer_thr, eer_val = find_eer_threshold(probs, labels)
    apcer_05, bpcer_05, hter_05 = compute_metrics_at_threshold(probs, labels, 0.5)
    apcer_eer, bpcer_eer, hter_eer = compute_metrics_at_threshold(probs, labels, eer_thr)
    return {
        'AUC': round(auc, 4),
        'EER_threshold': round(eer_thr, 4),
        'EER (%)': round(eer_val * 100, 4),
        'APCER@0.5 (%)': round(apcer_05, 4),
        'BPCER@0.5 (%)': round(bpcer_05, 4),
        'HTER@0.5 (%)': round(hter_05, 4),
        'APCER@EER (%)': round(apcer_eer, 4),
        'BPCER@EER (%)': round(bpcer_eer, 4),
        'HTER@EER (%)': round(hter_eer, 4),
    }


# ==========================================
# 5. ANA DÖNGÜ
# ==========================================
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"FPAD - YEPYENİ RESNET MODELLERİ İÇİN EER DEĞERLENDİRMESİ")
    print(f"Donanım: {device}")
    print(f"{'='*70}\n")

    # --- DİZİN YOLLARI ---
    MODEL_DIR = os.path.expanduser("~/Desktop/fpad_diffusion/results/yepyeni_resnet")
    OUTPUT_DIR = os.path.expanduser("~/Desktop/fpad_diffusion/results")
    OULU_TEST = os.path.expanduser("~/Desktop/fpad_diffusion/data/processed/OULU/test")
    REPLAY_TEST = os.path.expanduser("~/Desktop/fpad_diffusion/data/processed/ReplayAttack/test")

    print(f"Model klasörü: {MODEL_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Test setlerini yükle
    print("\nTest veri setleri yükleniyor...")
    oulu_ds = FlexibleFPADDataset(OULU_TEST, transform=test_transform)
    replay_ds = FlexibleFPADDataset(REPLAY_TEST, transform=test_transform)
    oulu_loader = DataLoader(oulu_ds, batch_size=32, shuffle=False, num_workers=4)
    replay_loader = DataLoader(replay_ds, batch_size=32, shuffle=False, num_workers=4)

    # Modelleri bul
    if not os.path.exists(MODEL_DIR):
        print(f"\nHATA: Klasör bulunamadı: {MODEL_DIR}")
        return

    pth_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')])
    print(f"\nTest edilecek model sayısı: {len(pth_files)}")
    for f in pth_files:
        print(f"  - {f}")

    if not pth_files:
        print("\nHATA: Klasörde .pth dosyası yok.")
        return

    all_results = []

    for model_name in pth_files:
        model_path = os.path.join(MODEL_DIR, model_name)
        print(f"\n{'─'*70}")
        print(f"Model: {model_name}")
        print(f"{'─'*70}")

        try:
            model = load_model(model_path, device)
        except Exception as e:
            print(f"  HATA - model yüklenemedi: {e}")
            continue

        # Eğitim setini belirle
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
            print(f"  [SKIP] Bilinmeyen model türü")
            continue

        # INTRA
        print(f"  [Intra]  {train_set} → {train_set}")
        intra_probs, intra_labels = collect_predictions(model, intra_loader, device)
        intra_results = evaluate_with_both_thresholds(intra_probs, intra_labels)

        # CROSS
        print(f"  [Cross]  {train_set} → {cross_set}")
        cross_probs, cross_labels = collect_predictions(model, cross_loader, device)
        cross_results = evaluate_with_both_thresholds(cross_probs, cross_labels)

        print(f"\n  INTRA AUC: {intra_results['AUC']:.4f} | "
              f"HTER@EER: %{intra_results['HTER@EER (%)']:.2f}")
        print(f"  CROSS AUC: {cross_results['AUC']:.4f} | "
              f"HTER@EER: %{cross_results['HTER@EER (%)']:.2f}")

        all_results.append({
            'model': model_name,
            'train_set': train_set,
            'cross_set': cross_set,
            'intra': intra_results,
            'cross': cross_results,
        })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_results(all_results, OUTPUT_DIR)


def write_results(all_results, output_dir):
    """3 dosya yaz: detaylı intra, detaylı cross, özet karşılaştırma."""

    # --- 1) DETAYLI INTRA SONUÇLAR ---
    intra_path = os.path.join(output_dir, "intra_dataset_yepyeni_results.txt")
    with open(intra_path, 'w') as f:
        f.write("FPAD - YEPYENİ RESNET MODELLERİ INTRA-DATASET (Detaylı)\n")
        f.write("=" * 70 + "\n\n")
        for r in all_results:
            f.write(f"Model: {r['model']}\n")
            f.write(f"Test Seti: {r['train_set']}\n")
            res = r['intra']
            f.write(f"  AUC: {res['AUC']}\n")
            f.write(f"  EER Threshold: {res['EER_threshold']}\n")
            f.write(f"  EER: %{res['EER (%)']}\n")
            f.write(f"  --- Threshold = 0.5 ---\n")
            f.write(f"    APCER: %{res['APCER@0.5 (%)']}\n")
            f.write(f"    BPCER: %{res['BPCER@0.5 (%)']}\n")
            f.write(f"    HTER:  %{res['HTER@0.5 (%)']}\n")
            f.write(f"  --- Threshold = EER ---\n")
            f.write(f"    APCER: %{res['APCER@EER (%)']}\n")
            f.write(f"    BPCER: %{res['BPCER@EER (%)']}\n")
            f.write(f"    HTER:  %{res['HTER@EER (%)']}\n")
            f.write("-" * 70 + "\n")

    # --- 2) DETAYLI CROSS SONUÇLAR ---
    cross_path = os.path.join(output_dir, "cross_dataset_yepyeni_results.txt")
    with open(cross_path, 'w') as f:
        f.write("FPAD - YEPYENİ RESNET MODELLERİ CROSS-DATASET (Detaylı)\n")
        f.write("=" * 70 + "\n\n")
        for r in all_results:
            f.write(f"Model: {r['model']}\n")
            f.write(f"Eğitim Seti: {r['train_set']}  →  Test Seti: {r['cross_set']}\n")
            res = r['cross']
            f.write(f"  AUC: {res['AUC']}\n")
            f.write(f"  EER Threshold: {res['EER_threshold']}\n")
            f.write(f"  EER: %{res['EER (%)']}\n")
            f.write(f"  --- Threshold = 0.5 ---\n")
            f.write(f"    APCER: %{res['APCER@0.5 (%)']}\n")
            f.write(f"    BPCER: %{res['BPCER@0.5 (%)']}\n")
            f.write(f"    HTER:  %{res['HTER@0.5 (%)']}\n")
            f.write(f"  --- Threshold = EER ---\n")
            f.write(f"    APCER: %{res['APCER@EER (%)']}\n")
            f.write(f"    BPCER: %{res['BPCER@EER (%)']}\n")
            f.write(f"    HTER:  %{res['HTER@EER (%)']}\n")
            f.write("-" * 70 + "\n")

    # --- 3) ÖZET TABLO ---
    summary_path = os.path.join(output_dir, "yepyeni_comparison_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("FPAD - YEPYENİ RESNET MODELLERİ - ÖZET\n")
        f.write("=" * 95 + "\n\n")

        # CROSS
        f.write(">>> CROSS-DATASET (Çapraz Test) — TEZ İÇİN ASIL TABLO <<<\n")
        f.write("-" * 95 + "\n")
        f.write(f"{'Model':<48} {'AUC':>7}  {'HTER@0.5':>10}  {'HTER@EER':>10}\n")
        f.write("-" * 95 + "\n")
        f.write("\n[OULU-NPU → Replay-Attack]\n")
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

        # INTRA
        f.write("\n\n>>> INTRA-DATASET (Aynı Veri Seti) <<<\n")
        f.write("-" * 95 + "\n")
        f.write(f"{'Model':<48} {'AUC':>7}  {'HTER@0.5':>10}  {'HTER@EER':>10}\n")
        f.write("-" * 95 + "\n")
        f.write("\n[OULU-NPU]\n")
        for r in all_results:
            if 'oulu' in r['model'].lower():
                res = r['intra']
                f.write(f"{r['model']:<48} {res['AUC']:>7.3f}  "
                        f"%{res['HTER@0.5 (%)']:>8.2f}  "
                        f"%{res['HTER@EER (%)']:>8.2f}\n")
        f.write("\n[Replay-Attack]\n")
        for r in all_results:
            if 'replay' in r['model'].lower():
                res = r['intra']
                f.write(f"{r['model']:<48} {res['AUC']:>7.3f}  "
                        f"%{res['HTER@0.5 (%)']:>8.2f}  "
                        f"%{res['HTER@EER (%)']:>8.2f}\n")

        # ESKİ MODELLERLE KARŞILAŞTIRMA
        f.write("\n\n" + "=" * 95 + "\n")
        f.write("ESKİ MODELLERLE KARŞILAŞTIRMA (referans)\n")
        f.write("=" * 95 + "\n")
        f.write("\nEski 8 modelin cross-dataset sonuçları (önceki comparison_summary.txt'den):\n\n")
        f.write("[OULU → Replay-Attack]\n")
        f.write(f"{'Model':<48} {'AUC':>7}  {'HTER@EER':>10}\n")
        f.write(f"{'resnet18_oulu_baseline_best (eski)':<48} {0.668:>7.3f}  %{33.97:>8.2f}\n")
        f.write(f"{'resnet18_oulu_gan_augmented_best (eski)':<48} {0.599:>7.3f}  %{39.23:>8.2f}\n")
        f.write(f"{'resnet18_oulu_augmented_final (eski 50K DDPM)':<48} {0.545:>7.3f}  %{44.95:>8.2f}\n")
        f.write(f"{'resnet18_oulu_ultimate_s2s_best (eski)':<48} {0.565:>7.3f}  %{43.83:>8.2f}\n")
        f.write("\n[Replay-Attack → OULU]\n")
        f.write(f"{'Model':<48} {'AUC':>7}  {'HTER@EER':>10}\n")
        f.write(f"{'resnet18_replay_baseline_best (eski)':<48} {0.580:>7.3f}  %{44.18:>8.2f}\n")
        f.write(f"{'resnet18_replay_gan_augmented_best (eski)':<48} {0.611:>7.3f}  %{42.27:>8.2f}\n")
        f.write(f"{'resnet18_replay_augmented_final (eski 50K DDPM)':<48} {0.576:>7.3f}  %{45.40:>8.2f}\n")
        f.write(f"{'resnet18_replay_ultimate_s2s_best (eski)':<48} {0.569:>7.3f}  %{46.30:>8.2f}\n")

        f.write("\n" + "=" * 95 + "\n")
        f.write("DEĞERLENDİRME REHBERİ:\n")
        f.write("- Eğer yepyeni baseline ~ eski baseline ise: training pipeline tutarlı çalışıyor\n")
        f.write("- Eğer yepyeni augmented modeller eski 50K modellerden iyiyse: doz optimizasyonu işe yaradı\n")
        f.write("- Eğer yepyeni DDPM > yepyeni GAN ise: hipotez doğrulandı (DDPM > GAN)\n")
        f.write("- Eğer yepyeni DDPM+S2S > yepyeni DDPM ise: S2S katkı sağladı\n")

    print(f"\n{'='*70}")
    print(f"✅ SONUÇLAR KAYDEDİLDİ")
    print(f"{'='*70}")
    print(f"  1. {intra_path}")
    print(f"  2. {cross_path}")
    print(f"  3. {summary_path}   ← TEZ İÇİN BU TABLO")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
