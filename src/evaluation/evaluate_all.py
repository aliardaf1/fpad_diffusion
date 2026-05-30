import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
from tqdm import tqdm

# ==========================================
# 1. ESNEK VERİ OKUMA SINIFI (OULU ve REPLAY İçin)
# ==========================================
class FlexibleFPADDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Hem Replay-Attack'in düz klasör yapısını hem de OULU-NPU'nun
        derinlemesine (iç içe type_X klasörleri olan) yapısını okur.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Desteklenen resim formatları
        valid_extensions = ('.png', '.jpg', '.jpeg')

        # --- GERÇEK (BONAFIDE - 1.0) VERİLERİNİ TOPLA ---
        real_base = os.path.join(root_dir, 'real')
        if os.path.exists(real_base):
            # os.walk kullanarak 'real' klasörünün içindeki tüm alt klasörleri dolaşır
            for root, dirs, files in os.walk(real_base):
                for file in files:
                    if file.lower().endswith(valid_extensions):
                        self.image_paths.append(os.path.join(root, file))
                        self.labels.append(1.0)
        else:
            print(f"Uyarı: {real_base} dizini bulunamadı!")

        # --- SAHTE (ATTACK - 0.0) VERİLERİNİ TOPLA ---
        attack_base = os.path.join(root_dir, 'attack')
        if os.path.exists(attack_base):
             # os.walk kullanarak 'attack' klasörünün içindeki tüm alt klasörleri (type_2, type_3 vb.) dolaşır
            for root, dirs, files in os.walk(attack_base):
                for file in files:
                    if file.lower().endswith(valid_extensions):
                        self.image_paths.append(os.path.join(root, file))
                        self.labels.append(0.0)
        else:
             print(f"Uyarı: {attack_base} dizini bulunamadı!")
             
        print(f"[{os.path.basename(root_dir)}] Yüklendi -> Gerçek: {self.labels.count(1.0)}, Sahte: {self.labels.count(0.0)}")

    def __len__(self): 
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform: 
                img = self.transform(img)
            return img, torch.tensor(self.labels[idx], dtype=torch.float32)
        except Exception as e:
            # Bozuk veya okunamayan bir dosya varsa hata fırlatmak yerine atla
            print(f"Hata okunamadı: {img_path} - {e}")
            # Basit bir siyah resim döndür (hata almamak için)
            dummy_img = torch.zeros(3, 256, 256)
            return dummy_img, torch.tensor(self.labels[idx], dtype=torch.float32)

# Normalizasyon
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 2. METRİK HESAPLAMA (Değişiklik Yok)
# ==========================================
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Test ediliyor", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs).squeeze(1)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).int()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    
    apcer = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    bpcer = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    hter = (apcer + bpcer) / 2.0
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    
    return {
        'APCER (%)': round(apcer * 100, 4),
        'BPCER (%)': round(bpcer * 100, 4),
        'HTER (%)': round(hter * 100, 4),
        'AUC': round(auc, 4)
    }

# ==========================================
# 3. ANA DÖNGÜ
# ==========================================
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Donanım: {device}")

    # --- DİZİN YOLLARI ---
    MODEL_DIR = os.path.expanduser("~/Desktop/fpad_diffusion/Resnet18")
    OUTPUT_DIR = os.path.expanduser("~/Desktop/fpad_diffusion/results")
    
    OULU_TEST_ROOT = os.path.expanduser("~/Desktop/fpad_diffusion/data/processed/OULU/test")
    REPLAY_TEST_ROOT = os.path.expanduser("~/Desktop/fpad_diffusion/data/processed/ReplayAttack/test")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "intra_dataset_results.txt")
    
    # DataLoader'ları oluşturma (Yeni esnek sınıf ile)
    oulu_ds = FlexibleFPADDataset(OULU_TEST_ROOT, transform=test_transform)
    replay_ds = FlexibleFPADDataset(REPLAY_TEST_ROOT, transform=test_transform)
    
    oulu_loader = DataLoader(oulu_ds, batch_size=32, shuffle=False, num_workers=4)
    replay_loader = DataLoader(replay_ds, batch_size=32, shuffle=False, num_workers=4)

    pth_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')]
    pth_files.sort()

    with open(output_file, 'w') as f:
        f.write("FPAD PROJESİ - INTRA-DATASET TEST SONUÇLARI\n")
        f.write("="*50 + "\n")
        
        for model_name in pth_files:
            model_path = os.path.join(MODEL_DIR, model_name)
            
            # Model oluşturma
            model = models.resnet18(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 1)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            
            if "oulu" in model_name:
                test_loader = oulu_loader
                dataset_name = "OULU-NPU"
            elif "replay" in model_name:
                test_loader = replay_loader
                dataset_name = "Replay-Attack"
            else:
                continue
                
            print(f"\nDeğerlendiriliyor: {model_name}")
            results = evaluate_model(model, test_loader, device)
            
            f.write(f"Model: {model_name}\n")
            f.write(f"Test Seti: {dataset_name}\n")
            for metric, value in results.items():
                f.write(f"  - {metric}: {value}\n")
            f.write("-" * 50 + "\n")
            
    print(f"\n✅ Sonuçlar {output_file} konumuna kaydedildi.")

if __name__ == "__main__":
    main()
