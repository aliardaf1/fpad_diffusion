import os
import glob
import random
import torch
from diffusers import UNet2DModel, DDPMScheduler
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# --- 1. KONFİGÜRASYON VE KLASÖR YAPISI ---
DATASET_NAME = "OULU-NPU"  
# DİKKAT: Veri model ağırlıklarının yolunu kontrol et
MODEL_WEIGHTS_PATH = "saved_models/ddpm_unet_epoch_50.pth"

# KAYNAK: Önceki adımda ürettiğimiz 50k spoof verisi
INPUT_DIR = "/home/undergrad25_1/Desktop/fpad_diffusion/data/synthetic/OULU-NPU/DDPM_spoof/spoof"

# HEDEF: Yeni S2S (Spoof-to-Spoof) verilerinin kaydedileceği yer
OUTPUT_DIR = "/home/undergrad25_1/Desktop/fpad_diffusion/data/S2S/OULU-NPU"

NUM_IMAGES = 5000   # Üretilecek S2S miktarı
BATCH_SIZE = 16 
IMAGE_SIZE = 256
TRAIN_TIMESTEPS = 1000
RANDOM_SEED = 42

# --- 2. SPOOF-TO-SPOOF (S2S) DÖNÜŞÜM GÜCÜ ---
# STRENGTH: 0.0 = Görüntü hiç değişmez, 1.0 = Görüntü tamamen silinir sıfırdan üretilir.
# 0.5 ile %50 orijinal resmi korur, %50 yepyeni DDPM varyasyonu katarız.
STRENGTH = 0.5 

# --- DONANIM İZOLASYONU ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Titan V
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

def initialize_model():
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

    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device, weights_only=True))
    model.eval()

    scheduler = DDPMScheduler(num_train_timesteps=TRAIN_TIMESTEPS, beta_schedule="linear")
    return model, scheduler

def load_source_images(input_dir, num_samples):
    """50k'lık spoof havuzundan rastgele resimler seçer."""
    all_images = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png"))
    if len(all_images) == 0:
        raise ValueError(f"HATA: Kaynak klasörde görüntü bulunamadı! Yol: {input_dir}")
    
    selected_images = random.sample(all_images, min(num_samples, len(all_images)))
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) # DDPM modeli [-1, 1] aralığını bekler
    ])
    return selected_images, transform

def generate_s2s_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    existing_files = glob.glob(os.path.join(OUTPUT_DIR, "*.jpg"))
    images_generated = len(existing_files)
    
    if images_generated >= NUM_IMAGES:
        print(f"Hedeflenen {NUM_IMAGES} S2S görüntü zaten üretilmiş! İşlem durduruluyor.")
        return
        
    remaining_images = NUM_IMAGES - images_generated
    
    print(f"\n--- SPOOF-TO-SPOOF (S2S) ÜRETİMİ ---")
    print(f"Kaynak Veri: {INPUT_DIR}")
    print(f"Hedef Veri: {OUTPUT_DIR}")
    print(f"Değişim Gücü (Strength): {STRENGTH}")
    print(f"Üretilecek Görüntü: {remaining_images}")
    print(f"------------------------------------\n")

    model, scheduler = initialize_model()
    
    # Kaynak resimleri seç ve yükle
    source_paths, transform = load_source_images(INPUT_DIR, remaining_images)
    
    # Diffusers mantığında başlangıç adımını belirle
    init_timestep = int(TRAIN_TIMESTEPS * STRENGTH)
    scheduler.set_timesteps(TRAIN_TIMESTEPS)
    start_index = TRAIN_TIMESTEPS - init_timestep 
    t_start = scheduler.timesteps[start_index]

    with torch.amp.autocast('cuda'), torch.no_grad():
        for i in range(0, remaining_images, BATCH_SIZE):
            batch_paths = source_paths[i : i + BATCH_SIZE]
            current_batch_size = len(batch_paths)
            
            # Resimleri Diskten Oku ve Tensore Çevir
            batch_images = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
            input_batch = torch.stack(batch_images).to(device)

            # S2S İşlemi: Orijinal resme t_start kadar gürültü ekle
            noise = torch.randn_like(input_batch)
            timesteps_tensor = torch.full((current_batch_size,), t_start, device=device, dtype=torch.long)
            noisy_image = scheduler.add_noise(input_batch, noise, timesteps_tensor)
            
            image = noisy_image
            
            # Eklenen gürültüyü geriye doğru temizleyerek yeni S2S görüntüyü oluştur
            for t in tqdm(scheduler.timesteps[start_index:], desc=f"Batch {i//BATCH_SIZE + 1} Üretiliyor", leave=False):
                residual = model(image, t).sample
                image = scheduler.step(residual, t, image).prev_sample
            
            # Görüntüleri Diske Kaydet
            for j in range(current_batch_size):
                file_path = os.path.join(OUTPUT_DIR, f"s2s_{DATASET_NAME}_{images_generated:05d}.jpg")
                save_image(image[j], file_path, normalize=True, value_range=(-1, 1))
                images_generated += 1
                
    print(f"\n{NUM_IMAGES} adet Spoof-to-Spoof görüntü başarıyla üretildi!")

if __name__ == "__main__":
    generate_s2s_images()
