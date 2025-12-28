import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers import UNet2DModel, DDPMScheduler
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset

# ==========================================
# 1. KONFİGÜRASYON VE HİPERPARAMETRELER
# ==========================================
class Config:
    image_size = 256          # Görüntü boyutu (256x256)
    batch_size = 16           # Mini-batch boyutu
    num_epochs = 50           # Toplam epoch sayısı
    learning_rate = 1e-4      # Öğrenme hızı
    num_timesteps = 1000      # 
    mixed_precision = "fp16"  # Hızlandırma için
    
    # Yollar
    train_data_path = "./data/Replay-Attack/train/attack" # SADECE ORİJİNAL SALDIRI (SPOOF) VERİSİ
    output_dir = "./ddpm_outputs"

# ==========================================
# 2. VERİ YÜKLEYİCİ (DATASET)
# ==========================================
class SpoofDataset(Dataset):
    """
    Sadece 'attack' klasöründeki spoof görüntülerini yükler.
    DDPM eğitimi için etikete (label) ihtiyaç yoktur, sadece görüntü döner.
    """
    def __init__(self, folder_path, size=256):
        self.folder_path = folder_path
        self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                            if f.endswith(('.png', '.jpg', '.jpeg', '.avi'))] # .avi varsa frame olmalı
        
        # PSD Section 6.1: Resize ve Normalizasyon
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # [-1, 1] aralığına çeker (DDPM standardı)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img)
        except Exception as e:
            print(f"Hata: {path} okunamadı.")
            return torch.zeros((3, Config.image_size, Config.image_size))

# ==========================================
# 3. EĞİTİM FONKSİYONU
# ==========================================
def train_loop():
    # Cihaz ayarı (GPU varsa kullan)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Eğitim Cihazı: {device}")
    
    # Klasör kontrolü
    os.makedirs(Config.output_dir, exist_ok=True)

    # A. Veri Setini Yükle
    dataset = SpoofDataset(Config.train_data_path, size=Config.image_size)
    train_dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4)
    print(f"Dataset yüklendi. Toplam Spoof Görüntü: {len(dataset)}")

    # B. Modeli Oluştur (PSD Section 6.3: UNet-based) 
    model = UNet2DModel(
        sample_size=Config.image_size,
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
        ),
    ).to(device)

    # C. Noise Scheduler (PSD Section 6.3: Linear variance) 
    noise_scheduler = DDPMScheduler(num_train_timesteps=Config.num_timesteps, beta_schedule="linear")

    # D. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)
    
    # E. Ana Döngü
    for epoch in range(Config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{Config.num_epochs}")
        epoch_loss = 0

        for step, clean_images in enumerate(train_dataloader):
            clean_images = clean_images.to(device)
            bs = clean_images.shape[0]

            # 1. Gürültü Örnekle (Sample Noise)
            noise = torch.randn_like(clean_images).to(device)
            
            # 2. Rastgele Zaman Adımları Seç (Timesteps)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()

            # 3. Görüntüye Gürültü Ekle (Forward Diffusion Process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # 4. Model Tahmini (Predict the Noise)
            # Model, gürültülü resme bakıp üzerindeki gürültüyü tahmin etmeye çalışır.
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

            # 5. Loss Hesapla (PSD Section 6.3: MSE Loss) 
            loss = F.mse_loss(noise_pred, noise)
            
            # 6. Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.update(1)
            progress_bar.set_postfix({"Loss": loss.item()})
            epoch_loss += loss.item()

        # Her 10 epoch'ta bir modeli kaydet
        if (epoch + 1) % 10 == 0:
            model.save_pretrained(os.path.join(Config.output_dir, f"ddpm-epoch-{epoch+1}"))
            print(f"\nModel epoch {epoch+1} sonunda kaydedildi.")

    print("Eğitim tamamlandı!")

if __name__ == "__main__":
    # Gerekli kütüphane kontrolü
    try:
        import diffusers
    except ImportError:
        print("Lütfen 'diffusers' kütüphanesini kurun: pip install diffusers")
        exit()
        
    train_loop()