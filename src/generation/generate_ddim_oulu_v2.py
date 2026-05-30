import os
import glob
import torch
from diffusers import UNet2DModel, DDIMScheduler
from torchvision import transforms, models
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

# --- 1. DONANIM İZOLASYONU (En Tepeye Alındı!) ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # OULU için Titan V (0) kullanıyoruz
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. KONFİGÜRASYON VE KLASÖR YAPISI ---
DATASET_NAME = "OULU-NPU"  
MODEL_WEIGHTS_PATH = "saved_models/ddpm_oulu_unet_epoch_50.pth"

# V3 İSİMLENDİRMESİ (Sarı filtreli verilerle karışmaz)
OUTPUT_DIR = f"data/synthetic/{DATASET_NAME}/DDIM_spoof_v2/spoof"

NUM_IMAGES = 50000
BATCH_SIZE = 16 # Titan V için 16 güvenli
IMAGE_SIZE = 256
INFERENCE_STEPS = 100 # Optimum kalite
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# --- 3. DÜZELTME: TERS NORMALİZASYON (Un-Normalize) İŞLEMİ ---
# OULU Eğitiminde (Transform kısmında) kullanılan normalizasyonun tersi
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Matematiksel Ters Normalizasyon
unnormalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(mean, std)],
    std=[1/s for s in std]
)

def initialize_model():
    model = UNet2DModel(
        sample_size=IMAGE_SIZE,
        in_channels=3,
        out_channels=3,
        # ... Mimari Parametreleri (Değişmedi) ...
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
    
    print(f"\n--- V3 KUSURSUZ VE RENK TAMİRLİ ÜRETİM BİLGİSİ (DDIM) ---")
    print(f"Hedef Veri Seti: {DATASET_NAME} (v3 Klasöründe)")
    print(f"Cihaz: {device} (FP32, TERS Normalizasyon Uygulanıyor)")
    print(f"-------------------------------------------\n")

    # DİKKAT: autocast YOK! 
    with torch.no_grad():
        for i in range(batch_count):
            current_batch_size = min(BATCH_SIZE, NUM_IMAGES - images_generated)
            
            noise = torch.randn(current_batch_size, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
            image = noise
            
            for t in tqdm(scheduler.timesteps, desc=f"Batch {i+1}/{batch_count}", leave=False):
                residual = model(image, t).sample
                image = scheduler.step(residual, t, image).prev_sample
            
            for j in range(current_batch_size):
                # --- V3 KESİN ÇÖZÜM: TERS NORMALİZASYON DÖNGÜSÜ ---
                # 1. [-1, 1] aralığını [0, 1] aralığına çekiyoruz.
                img_01 = (image[j] / 2 + 0.5)

                # 2. Ters Normalizasyon uygulayarak 'gerçek' ten rengini geri getiriyoruz.
                img_final = unnormalize(img_01).clamp(0, 1)

                file_path = os.path.join(OUTPUT_DIR, f"synth_v3_{DATASET_NAME}_{images_generated:05d}.jpg")
                save_image(img_final, file_path) 
                images_generated += 1
                
    print(f"\n{NUM_IMAGES} adet kusursuz v3 görüntüsü başarıyla üretildi!")

if __name__ == "__main__":
    generate_synthetic_images()
