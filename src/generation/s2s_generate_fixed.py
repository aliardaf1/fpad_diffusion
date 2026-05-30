"""
S2S (SPOOF-TO-SPOOF) GENERATE - OULU İÇİN, TEMİZ DDPM
======================================================
Replay versiyonunun OULU'ya uyarlanmış hali.
Orijinal OULU spoof görüntülerinden varyasyon üretir.

NASIL ÇALIŞIR:
1. Orijinal OULU spoof yüklenir
2. Forward process ile yarı gürültü eklenir
3. DDIM ile geri açılır (50 step)
4. Sonuç: Aynı kimlikli, farklı texture/lighting varyasyonu

KULLANIM:
  GPU_ID değiştir, NUM_S2S_IMAGES ayarla

ÇIKTI:
  /home/.../data/S2S_fixed/OULU-NPU/*.png
"""

import os
import glob
import random
import torch
from diffusers import UNet2DModel, DDIMScheduler
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm


# ==========================================
# AYARLAR
# ==========================================
GPU_ID = "0" # GTX 1080 (veya boştaki GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

# Üretim parametreleri
NUM_S2S_IMAGES = 1000      # Replay'de 1000 yapmıştık, simetri için aynı
S2S_STRENGTH = 0.5
INFERENCE_STEPS = 50
BATCH_SIZE = 16

# OULU YOLLARI
DATASET_NAME = "OULU-NPU"
# OULU train attack yolu — eğer Desktop'ta yoksa /media/... kullan
SOURCE_DIR = "/home/undergrad25_1/Desktop/fpad_diffusion/data/processed/OULU/train/attack"
MODEL_WEIGHTS_PATH = "/home/undergrad25_1/Desktop/fpad_diffusion/saved_models_fixed/ddpm_oulu_fixed_unet_epoch_50_ema.pth"
OUTPUT_DIR = f"/home/undergrad25_1/Desktop/fpad_diffusion/data/S2S_fixed/{DATASET_NAME}"

IMAGE_SIZE = 256
TRAIN_TIMESTEPS = 1000
RANDOM_SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


def initialize_model():
    print(f"\n{'='*60}")
    print(f"S2S GENERATE - OULU - GPU {GPU_ID}")
    print(f"{'='*60}")
    print(f"Üretilecek S2S:   {NUM_S2S_IMAGES}")
    print(f"S2S strength:     {S2S_STRENGTH}")
    print(f"Kaynak: {SOURCE_DIR}")
    print(f"Çıktı:  {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    if not os.path.exists(MODEL_WEIGHTS_PATH):
        raise FileNotFoundError(f"Model bulunamadı: {MODEL_WEIGHTS_PATH}\n"
                                 f"OULU DDPM training tamamlandı mı?")

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

    scheduler = DDIMScheduler(
        num_train_timesteps=TRAIN_TIMESTEPS,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon"
    )

    return model, scheduler


def load_source_images():
    print("Kaynak OULU spoof görüntüleri taranıyor...")
    image_paths = []
    valid_ext = ('.png', '.jpg', '.jpeg')

    if not os.path.exists(SOURCE_DIR):
        raise RuntimeError(f"HATA: {SOURCE_DIR} bulunamadı!\n"
                            f"OULU train attack yolu doğru mu? "
                            f"Belki /media/undergrad25_1/Data/oulu/... olmalı")

    for root, _, files in os.walk(SOURCE_DIR):
        for f in files:
            if f.lower().endswith(valid_ext):
                image_paths.append(os.path.join(root, f))

    if len(image_paths) == 0:
        raise RuntimeError(f"HATA: {SOURCE_DIR} klasöründe görüntü yok!")

    print(f"  Bulundu: {len(image_paths)} kaynak görüntü")
    random.seed(RANDOM_SEED)
    random.shuffle(image_paths)
    return image_paths


def generate_s2s_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    existing_files = glob.glob(os.path.join(OUTPUT_DIR, "s2s_*.png"))
    images_generated = len(existing_files)

    if images_generated >= NUM_S2S_IMAGES:
        print(f"Hedef ({NUM_S2S_IMAGES}) zaten dolu, çıkılıyor.")
        return

    remaining = NUM_S2S_IMAGES - images_generated
    batch_count = (remaining + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Önceki:    {images_generated}")
    print(f"Kalan:     {remaining}")
    print(f"Batch:     {batch_count}\n")

    model, scheduler = initialize_model()
    source_paths = load_source_images()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    scheduler.set_timesteps(INFERENCE_STEPS)
    init_timestep = int(INFERENCE_STEPS * S2S_STRENGTH)
    t_start = max(INFERENCE_STEPS - init_timestep, 0)
    timesteps_to_use = scheduler.timesteps[t_start:]

    print(f"  S2S strength {S2S_STRENGTH} → {len(timesteps_to_use)} denoising step\n")

    with torch.no_grad():
        for batch_idx in range(batch_count):
            current_bs = min(BATCH_SIZE, NUM_S2S_IMAGES - images_generated)
            if current_bs <= 0:
                break

            batch_sources = []
            for _ in range(current_bs):
                src_path = random.choice(source_paths)
                src_img = Image.open(src_path).convert("RGB")
                src_tensor = transform(src_img)
                batch_sources.append(src_tensor)

            source_batch = torch.stack(batch_sources).to(device)

            # Forward: gürültü ekle
            noise = torch.randn_like(source_batch)
            timesteps_for_noise = torch.tensor([timesteps_to_use[0]] * current_bs,
                                                device=device).long()
            noisy_batch = scheduler.add_noise(source_batch, noise, timesteps_for_noise)

            # Reverse: denoise
            image = noisy_batch
            for t in tqdm(timesteps_to_use,
                          desc=f"S2S Batch {batch_idx+1}/{batch_count}",
                          leave=False, mininterval=3.0):
                residual = model(image, t).sample
                image = scheduler.step(residual, t, image).prev_sample

            for j in range(current_bs):
                file_path = os.path.join(
                    OUTPUT_DIR,
                    f"s2s_{DATASET_NAME}_{images_generated:05d}.png"
                )
                save_image(image[j], file_path, normalize=True, value_range=(-1, 1))
                images_generated += 1

            if (batch_idx + 1) % 5 == 0:
                print(f"  [+] {images_generated}/{NUM_S2S_IMAGES} S2S üretildi")

    print(f"\n{'='*60}")
    print(f"OULU S2S ÜRETİM TAMAMLANDI")
    print(f"Toplam: {images_generated} S2S görüntü")
    print(f"Konum:  {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    generate_s2s_images()
