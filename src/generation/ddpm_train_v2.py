"""
DDPM TRAINING - DÜZELTİLMİŞ VERSİYON
=======================================
Renk cast probleminin çözümü için yapılan değişiklikler:

1. FP16 AUTOCAST KALDIRILDI → FP32 training (kanal cast'in ana sebebi)
2. LINEAR → COSINE BETA SCHEDULE (squaredcos_cap_v2, 256x256 için standart)
3. EMA EKLENDİ (decay=0.9999, modern DDPM training'in olmazsa olmazı)
4. MIN-SNR-γ LOSS WEIGHTING (γ=5, bazı timestep'lerde renk drift'i önler)
5. LEARNING RATE SCHEDULER (cosine annealing)
6. GRADIENT CLIPPING (max_norm=1.0)
7. ARA SAMPLE ÜRETİMİ (her 5 epoch'ta kontrol görüntüsü)
8. EMA ağırlıkları AYRI KAYDEDİLİR (.pth ve _ema.pth)

KULLANIM:
  # OULU için
  ACTIVE_DATASET = "OULU" ayarla, çalıştır
  # REPLAY için
  ACTIVE_DATASET = "REPLAY" ayarla, çalıştır

ÇIKTI:
  - {prefix}_unet_epoch_N.pth        (normal ağırlıklar)
  - {prefix}_unet_epoch_N_ema.pth    (EMA ağırlıkları - INFERENCE'TE KULLAN!)
  - {prefix}_samples_epoch_N.png     (her 5 epoch'ta ara örnekler)
"""

import os
import glob
import copy
import math
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


# ==========================================
# 1. DONANIM
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tekrarlanabilirlik
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==========================================
# 2. VERİ SETİ KONFİGÜRASYONU
# ==========================================
ACTIVE_DATASET = "OULU"  # "OULU" veya "REPLAY"

SAVE_DIR = "/home/undergrad25_1/Desktop/fpad_diffusion/saved_models_fixed"

if ACTIVE_DATASET == "OULU":
    DATASET_PATH = "/home/undergrad25_1/Desktop/fpad_diffusion/data/processed/OULU/train/attack"
    MODEL_PREFIX = "ddpm_oulu_fixed"
elif ACTIVE_DATASET == "REPLAY":
    DATASET_PATH = "/home/undergrad25_1/Desktop/fpad_diffusion/data/processed/ReplayAttack/train/attack"
    MODEL_PREFIX = "ddpm_replay_fixed"
else:
    raise ValueError("Geçersiz veri seti!")


# ==========================================
# 3. HİPERPARAMETRELER
# ==========================================
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-4
IMAGE_SIZE = 256
TIMESTEPS = 1000

# Yeni eklenenler:
EMA_DECAY = 0.9999          # EMA decay rate
MIN_SNR_GAMMA = 5.0         # Min-SNR-γ weighting
GRAD_CLIP_MAX_NORM = 1.0    # Gradient clipping
SAMPLE_EVERY_N_EPOCHS = 5   # Her 5 epoch'ta ara örnek
SAMPLE_BATCH_SIZE = 4       # Ara örnek için kaç görüntü

print(f"\n{'='*60}")
print(f"DDPM TRAINING - DÜZELTİLMİŞ VERSİYON")
print(f"{'='*60}")
print(f"Veri Seti: {ACTIVE_DATASET}")
print(f"Aktif Cihaz: {device}")
print(f"Veri Yolu:  {DATASET_PATH}")
print(f"Kayıt Yolu: {SAVE_DIR}")
print(f"\nKritik Değişiklikler:")
print(f"  - PRECISION: FP32 (autocast YOK)")
print(f"  - SCHEDULER: cosine (squaredcos_cap_v2)")
print(f"  - EMA:       decay={EMA_DECAY}")
print(f"  - LOSS:      Min-SNR-γ (γ={MIN_SNR_GAMMA})")
print(f"  - LR SCHED:  CosineAnnealingLR")
print(f"  - GRAD CLIP: {GRAD_CLIP_MAX_NORM}")
print(f"{'='*60}\n")


# ==========================================
# 4. VERİ SETİ SINIFI
# ==========================================
class LivenessDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # JPG ve PNG ikisini de oku
        search_paths = [
            os.path.join(root_dir, "**", "*.jpg"),
            os.path.join(root_dir, "**", "*.jpeg"),
            os.path.join(root_dir, "**", "*.png"),
        ]
        self.image_paths = []
        for sp in search_paths:
            self.image_paths.extend(glob.glob(sp, recursive=True))

        if len(self.image_paths) == 0:
            raise RuntimeError(f"HATA: '{root_dir}' yolunda hiç görüntü yok!")

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
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1] aralığı
])

dataset = LivenessDataset(root_dir=DATASET_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=4, pin_memory=True, drop_last=True)


# ==========================================
# 5. MODEL
# ==========================================
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


# ==========================================
# 6. NOISE SCHEDULER — COSINE
# ==========================================
# DEĞIŞIKLIK: linear → squaredcos_cap_v2 (cosine schedule)
# Bu, Improved DDPM paper'da (Nichol & Dhariwal 2021) 256x256+
# çözünürlükler için ÖNERILEN schedule.
noise_scheduler = DDPMScheduler(
    num_train_timesteps=TIMESTEPS,
    beta_schedule="squaredcos_cap_v2",  # Cosine schedule
    prediction_type="epsilon"            # Standart noise prediction
)


# ==========================================
# 7. OPTIMIZER + SCHEDULER
# ==========================================
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
# AdamW default weight_decay=0.01 idi. DDPM için 0 daha güvenli.

# Cosine annealing - learning rate cosine eğrisi ile düşer
total_steps = EPOCHS * len(dataloader)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)


# ==========================================
# 8. EMA — Exponential Moving Average
# ==========================================
class EMA:
    """
    Modern DDPM training'in OLMAZSA OLMAZI.
    Model ağırlıklarının hareketli ortalamasını tutar.
    Inference EMA ağırlıklarıyla yapılır → çok daha temiz görüntüler.
    """
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        # EMA ayrı bir model kopyası
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        """Her training step'inden sonra çağrılır"""
        for ema_p, model_p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)
        # Buffer'ları (BN stats vb.) da kopyala
        for ema_b, model_b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.data.copy_(model_b.data)

    def state_dict(self):
        return self.ema_model.state_dict()

ema = EMA(model, decay=EMA_DECAY)


# ==========================================
# 9. MIN-SNR-γ LOSS WEIGHTING
# ==========================================
def compute_snr(scheduler, timesteps):
    """
    SNR (Signal-to-Noise Ratio) hesaplama.
    Her timestep için sinyalin gürültüye oranı.
    """
    alphas_cumprod = scheduler.alphas_cumprod.to(timesteps.device)
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    alpha = sqrt_alphas_cumprod[timesteps]
    sigma = sqrt_one_minus_alphas_cumprod[timesteps]
    snr = (alpha / sigma) ** 2
    return snr


def min_snr_weights(scheduler, timesteps, gamma=5.0):
    """
    Min-SNR-γ loss weighting.
    Her timestep için loss'a uygulanacak ağırlık.
    Yüksek SNR timestep'lerde (az gürültü) loss azaltılır,
    böylece model orta-yüksek gürültü seviyelerinde DAHA İYİ öğrenir.
    Bu, renk drift'inin asıl sebebi olan timestep'leri dengeler.
    """
    snr = compute_snr(scheduler, timesteps)
    # min(SNR, γ) / SNR
    weights = torch.minimum(snr, torch.full_like(snr, gamma)) / snr
    return weights


# ==========================================
# 10. ARA SAMPLE ÜRETİMİ (Kontrol İçin)
# ==========================================
@torch.no_grad()
def generate_samples(ema_model, save_path, num_samples=4, ddim_steps=50):
    """
    Eğitim sırasında her N epoch'ta ara örnek üretir.
    EMA ağırlıkları + DDIM 50 step ile.
    """
    ema_model.eval()
    ddim_scheduler = DDIMScheduler(
        num_train_timesteps=TIMESTEPS,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon"
    )
    ddim_scheduler.set_timesteps(ddim_steps)

    # FP32'de inference (autocast YOK)
    noise = torch.randn(num_samples, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    image = noise

    for t in ddim_scheduler.timesteps:
        residual = ema_model(image, t).sample
        image = ddim_scheduler.step(residual, t, image).prev_sample

    # PNG kaydet
    save_image(image, save_path, normalize=True, value_range=(-1, 1), nrow=2)
    print(f"  [+] Ara örnek kaydedildi: {save_path}")


# ==========================================
# 11. AUTO-RESUME
# ==========================================
os.makedirs(SAVE_DIR, exist_ok=True)
start_epoch = 0

existing_models = glob.glob(os.path.join(SAVE_DIR, f"{MODEL_PREFIX}_unet_epoch_*.pth"))
# _ema.pth dosyalarını çıkar (sadece normal ağırlıklar bizi ilgilendiriyor)
existing_models = [m for m in existing_models if "_ema" not in m]

if existing_models:
    latest_model_path = max(existing_models,
                            key=lambda x: int(x.split('_epoch_')[-1].split('.pth')[0]))
    start_epoch = int(latest_model_path.split('_epoch_')[-1].split('.pth')[0])

    print(f"[*] CHECKPOINT BULUNDU: {latest_model_path}")
    print(f"[*] Epoch {start_epoch+1}'den devam ediliyor...")

    # Normal ağırlıkları yükle
    model.load_state_dict(torch.load(latest_model_path, map_location=device))

    # EMA ağırlıklarını yükle (varsa)
    ema_path = latest_model_path.replace('.pth', '_ema.pth')
    if os.path.exists(ema_path):
        ema.ema_model.load_state_dict(torch.load(ema_path, map_location=device))
        print(f"[*] EMA ağırlıkları da yüklendi.")
    else:
        print(f"[!] EMA dosyası yok, sıfırdan başlatıldı (normal ağırlıklardan).")
        ema = EMA(model, decay=EMA_DECAY)
else:
    print("[*] Checkpoint yok, 1. epoch'tan başlanıyor.\n")


# ==========================================
# 12. EĞİTİM DÖNGÜSÜ
# ==========================================
def train():
    print(f"\n{'='*60}")
    print(f"EĞİTİM BAŞLIYOR")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}",
                            mininterval=10.0)

        for batch in progress_bar:
            clean_images = batch.to(device)
            bs = clean_images.shape[0]

            # Gürültü ekleme (forward diffusion)
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                       (bs,), device=device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # ============================================
            # KRİTİK: FP32 forward (autocast YOK!)
            # ============================================
            noise_pred = model(noisy_images, timesteps).sample

            # ============================================
            # KRİTİK: Min-SNR-γ weighted MSE loss
            # ============================================
            # Her örnek için piksel-bazlı MSE (henüz reduce yok)
            loss_per_sample = F.mse_loss(noise_pred, noise, reduction='none')
            # [B, C, H, W] → [B] (per-sample loss)
            loss_per_sample = loss_per_sample.mean(dim=[1, 2, 3])

            # Min-SNR-γ ağırlıkları
            weights = min_snr_weights(noise_scheduler, timesteps, gamma=MIN_SNR_GAMMA)
            loss = (loss_per_sample * weights).mean()

            # Backward + step
            optimizer.zero_grad()
            loss.backward()

            # ============================================
            # KRİTİK: Gradient clipping
            # ============================================
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            max_norm=GRAD_CLIP_MAX_NORM)

            optimizer.step()
            lr_scheduler.step()  # Learning rate update

            # ============================================
            # KRİTİK: EMA update (her step'te)
            # ============================================
            ema.update(model)

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), lr=lr_scheduler.get_last_lr()[0])

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"\n---> Epoch {epoch+1} | Ort. Loss (MinSNR-MSE): {avg_epoch_loss:.4f} | "
              f"LR: {lr_scheduler.get_last_lr()[0]:.2e}\n")

        # ============================================
        # Her 5 epoch'ta kayıt + ara örnek
        # ============================================
        if (epoch + 1) % SAMPLE_EVERY_N_EPOCHS == 0 or (epoch + 1) == EPOCHS:
            # Normal ağırlıklar
            model_path = os.path.join(SAVE_DIR, f"{MODEL_PREFIX}_unet_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)

            # EMA ağırlıkları (INFERENCE'TE BUNU KULLAN)
            ema_path = os.path.join(SAVE_DIR, f"{MODEL_PREFIX}_unet_epoch_{epoch+1}_ema.pth")
            torch.save(ema.state_dict(), ema_path)

            print(f"[+] Modeller kaydedildi:")
            print(f"    Normal: {os.path.basename(model_path)}")
            print(f"    EMA:    {os.path.basename(ema_path)} ← INFERENCE'TE BU\n")

            # Ara örnek üret (kalite kontrolü)
            sample_path = os.path.join(SAVE_DIR,
                                        f"{MODEL_PREFIX}_samples_epoch_{epoch+1}.png")
            try:
                generate_samples(ema.ema_model, sample_path,
                                 num_samples=SAMPLE_BATCH_SIZE)
            except Exception as e:
                print(f"  [!] Ara örnek üretiminde hata: {e}")

    print(f"\n{'='*60}")
    print(f"EĞİTİM TAMAMLANDI")
    print(f"{'='*60}")
    print(f"\nSONRAKİ ADIM:")
    print(f"  ddpm_generate_fixed.py kullan, EMA ağırlığını seç:")
    print(f"  MODEL_WEIGHTS_PATH = '{MODEL_PREFIX}_unet_epoch_50_ema.pth'")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    train()
