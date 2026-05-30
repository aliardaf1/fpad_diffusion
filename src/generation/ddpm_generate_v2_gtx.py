"""
DDIM GENERATE - TRAINING SAMPLE İLE BIREBIR AYNI SETUP
========================================================
Training kodundaki generate_samples() fonksiyonu DDIM kullanıyordu ve
oradaki örnekler TEMİZdi. Aynı setup'ı tam taklit ediyoruz.

DEĞIŞIKLIK (DDPM yerine DDIM):
- DDPMScheduler → DDIMScheduler
- 1000 step → 50 step (DDIM ile yeter)
- DDPM 1000 step'inin biriktirdiği gürültü drift'i ortadan kalkar

DİĞER HER ŞEY AYNI:
- FP32 (autocast YOK)
- EMA ağırlıkları
- Cosine schedule (squaredcos_cap_v2)
- PNG kayıt
- prediction_type="epsilon"

İKİ GPU PARALEL ÇALIŞTIRMA:
  Kopya 1: GPU_ID="0", NUM_IMAGES=8000
  Kopya 2: GPU_ID="1", NUM_IMAGES=2000
  Aynı klasöre yazarlar, çakışma yok (gpu_id dosya isminde)
"""

import os
GPU_ID = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

import glob
import torch
from diffusers import UNet2DModel, DDIMScheduler
from torchvision.utils import save_image
from tqdm import tqdm


# ==========================================
# DEĞIŞTIRILECEK AYARLAR
# ==========================================
        # "0" (Titan V) veya "1" (GTX 1080) YUKARIDA.
NUM_IMAGES = 2000     # Bu GPU kaç görüntü üretecek


# ==========================================
# KONFİGÜRASYON
# ==========================================
ACTIVE_DATASET = "OULU"  # "OULU" veya "REPLAY"


if ACTIVE_DATASET == "OULU":
    DATASET_NAME = "OULU-NPU"
    MODEL_WEIGHTS_PATH = "/home/undergrad25_1/Desktop/fpad_diffusion/saved_models_fixed/ddpm_oulu_fixed_unet_epoch_50_ema.pth"
    OUTPUT_DIR = f"/home/undergrad25_1/Desktop/fpad_diffusion/data/synthetic/{DATASET_NAME}/DDPM_spoof_fixed/spoof"
elif ACTIVE_DATASET == "REPLAY":
    DATASET_NAME = "ReplayAttack"
    MODEL_WEIGHTS_PATH = "/home/undergrad25_1/Desktop/fpad_diffusion/saved_models_fixed/ddpm_replay_fixed_unet_epoch_50_ema.pth"
    OUTPUT_DIR = f"/home/undergrad25_1/Desktop/fpad_diffusion/data/synthetic/{DATASET_NAME}/DDPM_spoof_fixed/spoof"
else:
    raise ValueError("Geçersiz veri seti!")

# Parametreler (training ile aynı)
BATCH_SIZE = 16
IMAGE_SIZE = 256
TRAIN_TIMESTEPS = 1000      # Modelin eğitildiği step sayısı
INFERENCE_STEPS = 50        # DDIM 50 step (training sample'da kullanılan)
RANDOM_SEED = 42 + int(GPU_ID)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ==========================================
# MODEL YÜKLEME
# ==========================================
def initialize_model():
    gpu_name = "Titan V" if GPU_ID == "0" else "GTX 1080"
    print(f"\n{'='*60}")
    print(f"DDIM GENERATE - GPU {GPU_ID} ({gpu_name})")
    print(f"{'='*60}")
    print(f"Veri Seti:        {DATASET_NAME}")
    print(f"Üretilecek sayı:  {NUM_IMAGES}")
    print(f"Batch size:       {BATCH_SIZE}")
    print(f"DDIM step:        {INFERENCE_STEPS}")
    print(f"\nTraining sample ile birebir aynı setup:")
    print(f"  - DDIMScheduler")
    print(f"  - {INFERENCE_STEPS} step")
    print(f"  - Cosine schedule (squaredcos_cap_v2)")
    print(f"  - FP32 (autocast YOK)")
    print(f"  - EMA ağırlıkları")
    print(f"{'='*60}\n")

    if not os.path.exists(MODEL_WEIGHTS_PATH):
        raise FileNotFoundError(f"Model bulunamadı: {MODEL_WEIGHTS_PATH}")

    # Model mimarisi (training ile birebir aynı)
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
        ),
    ).to(device)

    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device,
                                       weights_only=True))
    model.eval()

    # ============================================
    # KRİTİK: DDIMScheduler (training sample ile aynı)
    # ============================================
    scheduler = DDIMScheduler(
        num_train_timesteps=TRAIN_TIMESTEPS,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon"
    )

    return model, scheduler


# ==========================================
# GENERATE
# ==========================================
def generate_synthetic_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Sadece bu GPU'nun ürettiği dosyaları say (numaralandırma çakışmasın)
    existing_files = glob.glob(os.path.join(OUTPUT_DIR, f"synth_gpu{GPU_ID}_*.png"))
    existing_files += glob.glob(os.path.join(OUTPUT_DIR, f"synth_gpu{GPU_ID}_*.jpg"))
    images_generated = len(existing_files)

    if images_generated >= NUM_IMAGES:
        print(f"GPU {GPU_ID} hedefi ({NUM_IMAGES}) zaten dolu, çıkılıyor.")
        return

    remaining = NUM_IMAGES - images_generated
    batch_count = (remaining + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Önceki:           {images_generated}")
    print(f"Kalan:            {remaining}")
    print(f"Batch sayısı:     {batch_count}\n")

    model, scheduler = initialize_model()

    # FP32 inference (autocast YOK)
    with torch.no_grad():
        for i in range(batch_count):
            current_bs = min(BATCH_SIZE, NUM_IMAGES - images_generated)

            # Saf gürültüden başla
            noise = torch.randn(current_bs, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
            image = noise

            # ============================================
            # DDIM 50 step (DDPM 1000 step yerine)
            # ============================================
            scheduler.set_timesteps(INFERENCE_STEPS)

            for t in tqdm(scheduler.timesteps,
                          desc=f"GPU{GPU_ID} Batch {i+1}/{batch_count}",
                          leave=False, mininterval=5.0):
                residual = model(image, t).sample
                image = scheduler.step(residual, t, image).prev_sample

            # Kaydet (PNG)
            for j in range(current_bs):
                file_path = os.path.join(
                    OUTPUT_DIR,
                    f"synth_gpu{GPU_ID}_{DATASET_NAME}_{images_generated:05d}.png"
                )
                save_image(image[j], file_path, normalize=True, value_range=(-1, 1))
                images_generated += 1

            # İlerleme bildirimi
            if (i + 1) % 10 == 0:
                print(f"  [GPU{GPU_ID}] {images_generated}/{NUM_IMAGES} üretildi")

    print(f"\n{'='*60}")
    print(f"GPU {GPU_ID} ÜRETİM TAMAMLANDI")
    print(f"Toplam üretilen:  {images_generated}")
    print(f"Konum:            {OUTPUT_DIR}")
    print(f"{'='*60}")
    print(f"\nSONRAKİ ADIM:")
    print(f"  1. İlk birkaç görüntüyü incele - renk cast var mı?")
    print(f"  2. Cast YOKsa → tamamdır, generate'i tamamlamasını bekle")
    print(f"  3. Cast VARsa → bana göster, beraber bakalım")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    generate_synthetic_images()
