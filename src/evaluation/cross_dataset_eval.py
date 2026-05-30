import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

# Modüler İçe Aktarma: Aynı dizindeki evaluate_all.py dosyasından gerekli modülleri çeker
from evaluate_all import FlexibleFPADDataset, evaluate_model, test_transform

def run_cross_dataset_evaluation():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Çapraz Değerlendirme (Cross-Dataset) Başlıyor... Donanım: {device}")

    # --- DİZİN YOLLARI ---
    MODEL_DIR = os.path.expanduser("~/Desktop/fpad_diffusion/Resnet18")
    OUTPUT_DIR = os.path.expanduser("~/Desktop/fpad_diffusion/results")
    OULU_TEST_ROOT = os.path.expanduser("~/Desktop/fpad_diffusion/data/processed/OULU/test")
    REPLAY_TEST_ROOT = os.path.expanduser("~/Desktop/fpad_diffusion/data/processed/ReplayAttack/test")
    
    output_file = os.path.join(OUTPUT_DIR, "cross_dataset_results.txt")

    # DataLoader'ları Oluşturma
    print("Test veri setleri yükleniyor...")
    oulu_ds = FlexibleFPADDataset(OULU_TEST_ROOT, transform=test_transform)
    replay_ds = FlexibleFPADDataset(REPLAY_TEST_ROOT, transform=test_transform)
    
    oulu_loader = DataLoader(oulu_ds, batch_size=32, shuffle=False, num_workers=4)
    replay_loader = DataLoader(replay_ds, batch_size=32, shuffle=False, num_workers=4)

    # Modelleri Listele
    pth_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')])

    with open(output_file, 'w') as f:
        f.write("FPAD PROJESİ - CROSS-DATASET (ÇAPRAZ) TEST SONUÇLARI\n")
        f.write("="*50 + "\n")
        
        for model_name in pth_files:
            model_path = os.path.join(MODEL_DIR, model_name)
            
            # Model Mimarisini Kur
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 1)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            
            # ÇAPRAZ TEST MANTIĞI: Modelin eğitilmediği diğer veri setini seç
            if "oulu" in model_name:
                cross_test_loader = replay_loader
                target_dataset = "Replay-Attack"
                train_dataset = "OULU-NPU"
            elif "replay" in model_name:
                cross_test_loader = oulu_loader
                target_dataset = "OULU-NPU"
                train_dataset = "Replay-Attack"
            else:
                continue
                
            print(f"\nModel: {model_name}\nEğitim Seti: {train_dataset} -> TEST EDİLEN SET: {target_dataset}")
            
            # evaluate_all.py'den gelen fonksiyonu çağır
            results = evaluate_model(model, cross_test_loader, device)
            
            print(f"Sonuçlar: {results}")
            f.write(f"Model Dosyası: {model_name}\n")
            f.write(f"Eğitim Seti: {train_dataset} | Test Edilen Set: {target_dataset}\n")
            for metric, value in results.items():
                f.write(f"  - {metric}: {value}\n")
            f.write("-" * 50 + "\n")
            
    print(f"\n✅ İşlem Tamamlandı. Çapraz test sonuçları kaydedildi: {output_file}")

if __name__ == "__main__":
    run_cross_dataset_evaluation()
