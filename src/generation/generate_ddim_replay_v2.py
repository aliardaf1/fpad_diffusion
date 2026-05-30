import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1" #

import glob
import torch
from diffusers import UNet2DModel, DDIMScheduler
from torchvision.utils import save_image
from tqdm import tqdm

# --- 1. KONFİGÜRASYON VE KLASÖR YAPISI ---
DATASET_NAME = "ReplayAttack"  
MODEL_WEIGHTS_PATH = "saved_models/ddpm_unet_epoch_50.pth"

# V2 İSİMLENDİRMESİ (Eski verilerle ASLA karışmaz)
OUTPUT_DIR = f"data/synthetic/{DATASET_NAME}/DDIM_spoof_v2/spoof"

NUM_IMAGES = 50000
BATCH_SIZE = 8 
IMAGE_SIZE = 256
INFERENCE_STEPS = 100 # DDIM YAPAN KISIM BU.
RANDOM_SEED = 42

# --- DONANIM İZOLASYONU ---
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

    # DDIM Scheduler (Hızlı ve Kaliteli Üretim)
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    scheduler.set_timesteps(INFERENCE_STEPS) 
    
    return model, scheduler

def generate_synthetic_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    existing_files = glob.glob(os.path.join(OUTPUT_DIR, "*.jpg"))
    images_generated = len(existing_files)
    
    if images_generated >= NUM_IMAGES:
        print(f"Hedeflenen {NUM_IMAGES} görüntü zaten üretilmiş!")
        return
        
    remaining_images = NUM_IMAGES - images_generated
    batch_count = (remaining_images + BATCH_SIZE - 1) // BATCH_SIZE

    model, scheduler = initialize_model()
    
    print(f"\n--- V2 KUSURSUZ ÜRETİM BİLGİSİ (DDIM) ---")
    print(f"Hedef Veri Seti: {DATASET_NAME} (v2 Klasöründe)")
    print(f"Hedef: {NUM_IMAGES} Görüntü (Kalan: {remaining_images})")
    print(f"Zaman Adımı: {INFERENCE_STEPS} (Optimum Kalite & Hız)")
    print(f"Cihaz: {device}")
    print(f"-------------------------------------------\n")

    # DİKKAT: autocast YOK! Renklerin bozulmaması için FP32 kullanıyoruz.
    with torch.no_grad():
        for i in range(batch_count):
            current_batch_size = min(BATCH_SIZE, NUM_IMAGES - images_generated)
            
            noise = torch.randn(current_batch_size, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
            image = noise
            
            for t in tqdm(scheduler.timesteps, desc=f"Batch {i+1}/{batch_count}", leave=False):
                residual = model(image, t).sample
                image = scheduler.step(residual, t, image).prev_sample
            
            # --- RENK BOZULMASINI (NEON ARTIFACT) KESİN ÖNLEYEN KİLİT ---
            final_image = (image / 2 + 0.5).clamp(0, 1) 
            
            for j in range(current_batch_size):
                # Dosya isimlerine de "v2" ekledik
                file_path = os.path.join(OUTPUT_DIR, f"synth_v2_{DATASET_NAME}_{images_generated:05d}.jpg")
                save_image(final_image[j].float(), file_path) 
                images_generated += 1
                
    print(f"\n{NUM_IMAGES} adet kusursuz v2 görüntüsü başarıyla üretildi!")

if __name__ == "__main__":
    generate_synthetic_images()
