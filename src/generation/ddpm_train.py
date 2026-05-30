import os
import glob
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import amp

from diffusers import UNet2DModel, DDPMScheduler
from torch.optim import AdamW
from tqdm import tqdm

# --- 1. DONANIM İZOLASYONU ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. VERİ SETİ VE KAYIT (SAVE) DİZİNİ KONFİGÜRASYONU ---
# SADECE BURAYI DEĞİŞTİREREK İKİ VERİSETİ ARASINDA DEĞİŞİM YAPILABİLİR.
ACTIVE_DATASET = "OULU" 

# Modellerin kaydedileceği PROJE KLASÖR (Ana Diskiniz)
SAVE_DIR = "/home/undergrad25_1/Desktop/fpad_diffusion/saved_models"

if ACTIVE_DATASET == "OULU":
    # Verilerin okunacağı HARİCİ DİSK
    DATASET_PATH = "/media/undergrad25_1/Data/oulu/data_process/train/attack"
    MODEL_PREFIX = "ddpm_oulu"
elif ACTIVE_DATASET == "REPLAY":
    DATASET_PATH = "/home/undergrad25_1/Desktop/fpad_diffusion/data/processed/ReplayAttack/train/spoof"
    MODEL_PREFIX = "ddpm_replay"
else:
    raise ValueError("Geçersiz veri seti seçimi!")

# --- 3. HİPERPARAMETRELER ---
BATCH_SIZE = 4 
EPOCHS = 50
LEARNING_RATE = 1e-4
IMAGE_SIZE = 256
TIMESTEPS = 1000

print(f"\n--- SİSTEM BİLGİSİ ---")
print(f"Aktif Cihaz: {device}")
print(f"Veri Okuma: {DATASET_PATH}")
print(f"Model Kayıt: {SAVE_DIR}")
print(f"----------------------\n")

# --- 4. VERİ SETİ SINIFI ---
class LivenessDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        search_path = os.path.join(root_dir, "**", "*.jpg")
        self.image_paths = glob.glob(search_path, recursive=True)
        
        if len(self.image_paths) == 0:
            raise RuntimeError(f"HATA: '{search_path}' yolunda hiç görüntü bulunamadı!")
            
        print(f"[{ACTIVE_DATASET}] Toplam {len(self.image_paths)} görüntü yüklendi.\n")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
])

dataset = LivenessDataset(root_dir=DATASET_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# --- 5. MODEL VE SCHEDULER BAŞLATMA ---
model = UNet2DModel(
    sample_size=IMAGE_SIZE,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=(
        "DownBlock2D", "DownBlock2D", "DownBlock2D", 
        "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"
    ),
    up_block_types=(
        "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", 
        "UpBlock2D", "UpBlock2D", "UpBlock2D"
    )
).to(device)

noise_scheduler = DDPMScheduler(num_train_timesteps=TIMESTEPS)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# --- 6. AUTO-RESUME (Kaldığı Yerden Devam Etme Mantığı) ---
os.makedirs(SAVE_DIR, exist_ok=True)
start_epoch = 0

existing_models = glob.glob(os.path.join(SAVE_DIR, f"{MODEL_PREFIX}_unet_epoch_*.pth"))
if existing_models:
    # Dosya isimlerinden en büyük epoch numarasını bul
    latest_model_path = max(existing_models, key=lambda x: int(x.split('_epoch_')[-1].split('.pth')[0]))
    start_epoch = int(latest_model_path.split('_epoch_')[-1].split('.pth')[0])
    
    print(f"[*] CHECKPOINT BULUNDU: {latest_model_path}")
    print(f"[*] Ağırlıklar yükleniyor ve eğitime {start_epoch + 1}. Epoch'tan devam edilecek...\n")
    
    # Ağırlıkları modele yükle
    model.load_state_dict(torch.load(latest_model_path, map_location=device))
else:
    print("[*] Kayıtlı checkpoint bulunamadı. Eğitim 1. Epoch'tan sıfırdan başlıyor.\n")

# --- 7. EĞİTİM DÖNGÜSÜ ---
def train():
    scaler = amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # Döngü start_epoch'tan başlar (Eğer model 10. epoch'ta kaldıysa 10'dan başlar)
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0
        
        # Nohup kullanımında logların çok kirlenmemesi için mininterval=10 saniye yapıldı
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", mininterval=10.0)
        
        for batch in progress_bar:
            clean_images = batch.to(device)
            bs = clean_images.shape[0]
            
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            if device.type == 'cuda':
                with amp.autocast('cuda'):
                    noise_pred = model(noisy_images, timesteps).sample
                    loss = F.mse_loss(noise_pred, noise)
            else:
                noise_pred = model(noisy_images, timesteps).sample
                loss = F.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"\n---> Epoch {epoch+1} Özeti - Ortalama Kayıp (MSE): {avg_epoch_loss:.4f}\n")

        # Her 5 epoch'ta bir modeli Ana Diske kaydet
        if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS:
            model_save_path = os.path.join(SAVE_DIR, f"{MODEL_PREFIX}_unet_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"[+] Model başarıyla Ana Diske kaydedildi: {model_save_path}\n")

if __name__ == "__main__":
    train()
