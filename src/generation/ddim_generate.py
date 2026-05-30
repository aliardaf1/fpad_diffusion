import os
import torch
from diffusers import UNet2DModel, DDIMScheduler
from torchvision.utils import save_image
from tqdm import tqdm

# --- 1. KONFİGÜRASYON VE KLASÖR YAPISI ---
DATASET_NAME = "ReplayAttack"  
MODEL_WEIGHTS_PATH = "saved_models/ddpm_unet_epoch_50.pth"
OUTPUT_DIR = f"data/synthetic/{DATASET_NAME}/spoof"

NUM_IMAGES = 50000
BATCH_SIZE = 16 
IMAGE_SIZE = 256
TRAIN_TIMESTEPS = 1000
INFERENCE_STEPS = 50  # DDIM optimizasyonu

# --- DONANIM İZOLASYONU ---
# Üretim işlemini ana kartımız olan TITAN V'ye (GPU 0) yönlendiriyoruz
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Ağırlıkları güvenli modda yükle
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device, weights_only=True))
    model.eval()

    # DDIM Zamanlayıcısı
    scheduler = DDIMScheduler(num_train_timesteps=TRAIN_TIMESTEPS, beta_schedule="linear")
    return model, scheduler

def generate_synthetic_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model, scheduler = initialize_model()
    
    print(f"\n--- ÜRETİM BİLGİSİ ---")
    print(f"Hedef Veri Seti: {DATASET_NAME}")
    print(f"Üretilecek Miktar: {NUM_IMAGES} Görüntü")
    print(f"Adım Sayısı: {INFERENCE_STEPS} (DDIM Optimizasyonu)")
    print(f"Kullanılan Cihaz: {device} (TITAN V - GPU 0)")
    print(f"----------------------\n")

    images_generated = 0
    batch_count = (NUM_IMAGES + BATCH_SIZE - 1) // BATCH_SIZE

    with torch.amp.autocast('cuda'), torch.no_grad():
        for i in range(batch_count):
            current_batch_size = min(BATCH_SIZE, NUM_IMAGES - images_generated)
            noise = torch.randn(current_batch_size, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
            image = noise
            
            scheduler.set_timesteps(INFERENCE_STEPS)
            
            for t in tqdm(scheduler.timesteps, desc=f"Batch {i+1}/{batch_count} Üretiliyor", leave=False):
                residual = model(image, t).sample
                image = scheduler.step(residual, t, image).prev_sample
            
            for j in range(current_batch_size):
                file_path = os.path.join(OUTPUT_DIR, f"synth_{DATASET_NAME}_{images_generated:05d}.jpg")
                save_image(image[j], file_path, normalize=True, value_range=(-1, 1))
                images_generated += 1
                
        print(f"\n{NUM_IMAGES} adet görüntü başarıyla üretildi!")

if __name__ == "__main__":
    generate_synthetic_images()
