"""
FPAD - EER Threshold ile Yeniden Değerlendirme
================================================
Mevcut modelleri yeniden eğitmeden, threshold'ü cross-dataset için
kalibre ederek gerçek performansı ölçer.

Üretilen Çıktılar:
- intra_dataset_eer_results.txt
- cross_dataset_eer_results.txt
- comparison_summary.txt  (0.5 vs EER karşılaştırması)

Çalıştırma:
    python eer_threshold_eval.py
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
# 1. VERİ OKUMA SINIFI (Mevcut koddan birebir)
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
# 2. ÇIKARIM (INFERENCE) — Tüm probabiliteleri topla
# ==========================================
def collect_predictions(model, loader, device):
    """Model üzerinden tüm test setini geçirip probability ve label döndürür."""
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
# 3. METRİK HESAPLAMA — İki threshold ile
# ==========================================
def compute_metrics_at_threshold(probs, labels, threshold):
    """Verilen threshold'de APCER, BPCER, HTER hesaplar."""
    preds = (probs >= threshold).astype(int)
    # labels: 1=real, 0=attack
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    apcer = fp / (tn + fp) if (tn + fp) > 0 else 0.0  # Attack -> Real (yanlış kabul)
    bpcer = fn / (tp + fn) if (tp + fn) > 0 else 0.0  # Real -> Attack (yanlış red)
    hter = (apcer + bpcer) / 2.0
    return apcer * 100, bpcer * 100, hter * 100


def find_eer_threshold(probs, labels):
    """
    EER (Equal Error Rate) threshold'u bulur.
    EER: APCER == BPCER olduğu nokta.
    
    Not: sklearn'in roc_curve'ü 'pozitif sınıf'a göre çalışır.
    Bizde pozitif sınıf = real (1). 
    - TPR = real'in real olarak sınıflandırılma oranı = 1 - BPCER
    - FPR = attack'in real olarak sınıflandırılma oranı = APCER
    EER: FPR == 1 - TPR -> APCER == BPCER
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    fnr = 1 - tpr  # FNR = BPCER (real kaçırma)
    # |FPR - FNR| en küçük olduğu index
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer_threshold = thresholds[eer_idx]
    eer_value = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    return float(eer_threshold), float(eer_value)


def evaluate_with_both_thresholds(probs, labels):
    """Hem 0.5 hem EER threshold ile metrikleri döndürür."""
    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0
    eer_thr, eer_val = find_eer_threshold(probs, labels)

    apcer_05, bpcer_05, hter_05 = compute_metrics_at_threshold(probs, labels, 0.5)
    apcer_eer, bpcer_eer, hter_eer = compute_metrics_at_threshold(probs, labels, eer_thr)

    return {
        'AUC': round(auc, 4),
        'EER_threshold': round(eer_thr, 4),
        'EER (%)': round(eer_val * 100, 4),
        # Default threshold (0.5)
        'APCER@0.5 (%)': round(apcer_05, 4),
        'BPCER@0.5 (%)': round(bpcer_05, 4),
        'HTER@0.5 (%)': round(hter_05, 4),
        # EER threshold
        'APCER@EER (%)': round(apcer_eer, 4),
        'BPCER@EER (%)': round(bpcer_eer, 4),
        'HTER@EER (%)': round(hter_eer, 4),
    }


# ==========================================
# 4. ANA DÖNGÜ
# ==========================================
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"FPAD - EER THRESHOLD YENİDEN DEĞERLENDİRME")
    print(f"Donanım: {device}")
    print(f"{'='*70}\n")

    # --- DİZİN YOLLARI ---
    MODEL_DIR = os.path.expanduser("~/Desktop/fpad_diffusion/Resnet18")
    OUTPUT_DIR = os.path.expanduser("~/Desktop/fpad_diffusion/results")
    OULU_TEST = os.path.expanduser("~/Desktop/fpad_diffusion/data/processed/OULU/test")
    REPLAY_TEST = os.path.expanduser("~/Desktop/fpad_diffusion/data/processed/ReplayAttack/test")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- VERİ SETLERİNİ HAZIRLA (Tek seferde, bellekte) ---
    print("Test veri setleri yükleniyor...")
    oulu_ds = FlexibleFPADDataset(OULU_TEST, transform=test_transform)
    replay_ds = FlexibleFPADDataset(REPLAY_TEST, transform=test_transform)
    oulu_loader = DataLoader(oulu_ds, batch_size=32, shuffle=False, num_workers=4)
    replay_loader = DataLoader(replay_ds, batch_size=32, shuffle=False, num_workers=4)

    # --- MODELLERİ BUL ---
    pth_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')])
    print(f"\nBulunan model sayısı: {len(pth_files)}\n")

    # Sonuçları depola — sonra dosyaya yazılacak
    all_results = []  # her item: dict

    for model_name in pth_files:
        model_path = os.path.join(MODEL_DIR, model_name)
        print(f"\n{'─'*70}")
        print(f"Model: {model_name}")
        print(f"{'─'*70}")

        # Modeli yükle
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        # Modelin eğitim seti
        if "oulu" in model_name:
            train_set = "OULU-NPU"
            intra_loader = oulu_loader
            cross_loader = replay_loader
            cross_set = "Replay-Attack"
        elif "replay" in model_name:
            train_set = "Replay-Attack"
            intra_loader = replay_loader
            cross_loader = oulu_loader
            cross_set = "OULU-NPU"
        else:
            print(f"  [SKIP] Bilinmeyen model türü: {model_name}")
            continue

        # === INTRA-DATASET ===
        print(f"  [Intra]  {train_set} → {train_set}")
        intra_probs, intra_labels = collect_predictions(model, intra_loader, device)
        intra_results = evaluate_with_both_thresholds(intra_probs, intra_labels)

        # === CROSS-DATASET ===
        print(f"  [Cross]  {train_set} → {cross_set}")
        cross_probs, cross_labels = collect_predictions(model, cross_loader, device)
        cross_results = evaluate_with_both_thresholds(cross_probs, cross_labels)

        # Özet yazdır
        print(f"\n  INTRA  HTER:  @0.5 = %{intra_results['HTER@0.5 (%)']:6.2f}  |  "
              f"@EER = %{intra_results['HTER@EER (%)']:6.2f}  |  AUC = {intra_results['AUC']}")
        print(f"  CROSS  HTER:  @0.5 = %{cross_results['HTER@0.5 (%)']:6.2f}  |  "
              f"@EER = %{cross_results['HTER@EER (%)']:6.2f}  |  AUC = {cross_results['AUC']}")

        improvement = cross_results['HTER@0.5 (%)'] - cross_results['HTER@EER (%)']
        print(f"  CROSS  HTER iyileşmesi (sadece threshold ile): -{improvement:.2f} puan")

        all_results.append({
            'model': model_name,
            'train_set': train_set,
            'cross_set': cross_set,
            'intra': intra_results,
            'cross': cross_results,
        })

        # GPU temizliği
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ==========================================
    # 5. SONUÇLARI DOSYAYA YAZ
    # ==========================================
    write_results(all_results, OUTPUT_DIR)


def write_results(all_results, output_dir):
    """Sonuçları 3 dosyaya yazar: intra, cross, ve comparison summary."""

    # --- 1) INTRA-DATASET RAPORU ---
    intra_path = os.path.join(output_dir, "intra_dataset_eer_results.txt")
    with open(intra_path, 'w') as f:
        f.write("FPAD PROJESİ - INTRA-DATASET SONUÇLAR (0.5 ve EER threshold)\n")
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

    # --- 2) CROSS-DATASET RAPORU ---
    cross_path = os.path.join(output_dir, "cross_dataset_eer_results.txt")
    with open(cross_path, 'w') as f:
        f.write("FPAD PROJESİ - CROSS-DATASET SONUÇLAR (0.5 ve EER threshold)\n")
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

    # --- 3) KARŞILAŞTIRMA TABLOSU (Tez için kritik) ---
    summary_path = os.path.join(output_dir, "comparison_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("FPAD PROJESİ - 0.5 vs EER THRESHOLD KARŞILAŞTIRMA ÖZETİ\n")
        f.write("=" * 95 + "\n\n")

        # Header
        f.write(f"{'Model':<48} {'AUC':>6}  {'HTER@0.5':>10}  {'HTER@EER':>10}  {'Δ':>8}\n")
        f.write("-" * 95 + "\n")

        # CROSS-DATASET — en kritik tablo
        f.write("\n>>> CROSS-DATASET (Çapraz Test) <<<\n")
        f.write("-" * 95 + "\n")
        for r in all_results:
            res = r['cross']
            delta = res['HTER@0.5 (%)'] - res['HTER@EER (%)']
            f.write(f"{r['model']:<48} {res['AUC']:>6.3f}  "
                    f"%{res['HTER@0.5 (%)']:>8.2f}  "
                    f"%{res['HTER@EER (%)']:>8.2f}  "
                    f"{delta:>+7.2f}\n")

        # INTRA-DATASET
        f.write("\n>>> INTRA-DATASET (Aynı Veri Seti) <<<\n")
        f.write("-" * 95 + "\n")
        for r in all_results:
            res = r['intra']
            delta = res['HTER@0.5 (%)'] - res['HTER@EER (%)']
            f.write(f"{r['model']:<48} {res['AUC']:>6.3f}  "
                    f"%{res['HTER@0.5 (%)']:>8.2f}  "
                    f"%{res['HTER@EER (%)']:>8.2f}  "
                    f"{delta:>+7.2f}\n")

        f.write("\n" + "=" * 95 + "\n")
        f.write("YORUM:\n")
        f.write("- AUC: Threshold'dan bağımsız ayırt etme gücü. >0.7 iyi, >0.85 çok iyi.\n")
        f.write("- HTER@0.5: Naif threshold ile sonuç (raporladığınız orijinal değerler).\n")
        f.write("- HTER@EER: Test setine kalibre threshold ile sonuç (alt sınır).\n")
        f.write("- Δ: Sadece threshold ayarıyla elde edilen iyileşme.\n")
        f.write("\n")
        f.write("NOT: EER threshold test setinden hesaplanır — bu 'oracle' bir alt sınırdır.\n")
        f.write("Gerçek deployment için DEV setinden threshold çıkarılmalıdır. Ama tezde\n")
        f.write("'threshold kalibrasyonu yapılırsa potansiyel performans' göstergesi olarak\n")
        f.write("kullanılabilir.\n")

    print(f"\n{'='*70}")
    print(f"✅ TÜM SONUÇLAR KAYDEDİLDİ")
    print(f"{'='*70}")
    print(f"  1. {intra_path}")
    print(f"  2. {cross_path}")
    print(f"  3. {summary_path}   ← TEZ İÇİN BU TABLOYU KULLANIN")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
