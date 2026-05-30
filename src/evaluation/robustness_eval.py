import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
from tqdm import tqdm

# ==========================================
# 1. GÜRÜLTÜ (NOISE) EKLEME SINIFI
# ==========================================
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        # Tensöre standart sapması 0.05 olan rastgele Gauss gürültüsü ekler
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        return torch.clamp(noisy_tensor, 0., 1.) # Değerleri 0-1 arasında tutar

# ==========================================
# 2. ESNEK VERİ OKUMA SINIFI (Bağımsız)
# ==========================================
class FlexibleFPADDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        valid_extensions = ('.png', '.jpg', '.jpeg')

        real_base = os.path.join(root_dir, 'real')
        if os.path.exists(real_base):
            for root, dirs, files in os.walk(real_base):
                for file in files:
                    if file.lower().endswith(valid_extensions):
                        self.image_paths.append(os.path.join(root, file))
                        self.labels.append(1.0)

        attack_base = os.path.join(root_dir, 'attack')
        if os.path.exists(attack_base):
            for root, dirs, files in os.walk(attack_base):
                for file in files:
                    if file.lower().endswith(valid_extensions):
                        self.image_paths.append(os.path.join(root, file))
                        self.labels.append(0.0)

    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform: img = self.transform(img)
            return img, torch.tensor(self.labels[idx], dtype=torch.float32)
        except Exception:
            return torch.zeros(3, 256, 256), torch.tensor(self.labels[idx], dtype=torch.float32)

# ==========================================
# 3. BOZULMA (PERTURBATION) SENARYOLARI
# ==========================================
# A. Temiz Veri (Referans için)
transform_clean = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# B. Bulanıklık (Gaussian Blur) Eklenmiş Veri
transform_blur = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.GaussianBlur(kernel_size=7, sigma=(1.5, 2.0)), # Güçlü bir bulanıklık
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# C. Gürültü (Noise) Eklenmiş Veri
transform_noise = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    AddGaussianNoise(0., 0.1), # Yüksek kumlanma
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 4. METRİK HESAPLAMA
# ==========================================
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Test ediliyor", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze(1)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).int()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    apcer = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    bpcer = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    hter = (apcer + bpcer) / 2.0
    
    return round(hter * 100, 4)

# ==========================================
# 5. ANA DÖNGÜ
# ==========================================
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Sağlamlık (Robustness) Analizi Başlıyor. Donanım: {device}")

    MODEL_DIR = os.path.expanduser("~/Desktop/fpad_diffusion/Resnet18")
    OUTPUT_DIR = os.path.expanduser("~/Desktop/fpad_diffusion/results")
    OULU_TEST_ROOT = os.path.expanduser("~/Desktop/fpad_diffusion/data/processed/OULU/test")
    REPLAY_TEST_ROOT = os.path.expanduser("~/Desktop/fpad_diffusion/data/processed/ReplayAttack/test")
    
    output_file = os.path.join(OUTPUT_DIR, "robustness_results.txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    conditions = {
        "Clean (Temiz)": transform_clean,
        "Blur (Bulanık)": transform_blur,
        "Noise (Gürültülü)": transform_noise
    }

    pth_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')])

    with open(output_file, 'w') as f:
        f.write("FPAD PROJESİ - ROBUSTNESS (SAĞLAMLIK) TEST SONUÇLARI\n")
        f.write("="*60 + "\n")
        
        for model_name in pth_files:
            model_path = os.path.join(MODEL_DIR, model_name)
            
            # Model Yükleme
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 1)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            
            # Intra-dataset seçimi
            if "oulu" in model_name:
                test_root = OULU_TEST_ROOT
                dataset_name = "OULU-NPU"
            elif "replay" in model_name:
                test_root = REPLAY_TEST_ROOT
                dataset_name = "Replay-Attack"
            else:
                continue
                
            print(f"\nModel: {model_name} ({dataset_name} üzerinde test ediliyor)")
            f.write(f"Model: {model_name}\n")
            f.write(f"Veri Seti: {dataset_name}\n")
            
            # Her bir bozulma koşulu için testi çalıştır
            for cond_name, transform in conditions.items():
                dataset = FlexibleFPADDataset(test_root, transform=transform)
                loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
                
                hter_val = evaluate_model(model, loader, device)
                print(f"  -> Koşul: {cond_name} | HTER: %{hter_val}")
                f.write(f"  - {cond_name} HTER: %{hter_val}\n")
                
            f.write("-" * 60 + "\n")
            
    print(f"\n✅ İşlem Tamamlandı. Sağlamlık test sonuçları kaydedildi: {output_file}")

if __name__ == "__main__":
    main()
