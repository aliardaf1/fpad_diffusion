import os
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import sys

# --- 1. KONFİGÜRASYON ---
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

DATASET_NAME = "OULU-NPU"
MODEL_WEIGHTS_PATH = "/home/undergrad25_1/Desktop/fpad_diffusion/GAN/oulu/latest_net_G.pth"
INPUT_DIR = "/home/undergrad25_1/Desktop/fpad_diffusion/data/processed/OULU/train/real/real"
OUTPUT_DIR = "/home/undergrad25_1/Desktop/fpad_diffusion/data/gan_synthetic_oulu"

TARGET_NUM_IMAGES = 50000 # Kesin hedef
BATCH_SIZE = 16
IMAGE_SIZE = 256
NUM_WORKERS = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. MİMARİ (ResNet 9-Blocks) ---
class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim), nn.ReLU(True), nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0), nn.InstanceNorm2d(dim)
        )
    def forward(self, x): return x + self.conv_block(x)

class ResNetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
        super(ResNetGenerator, self).__init__()
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        for i in range(2):
            mult = 2**i
            model += [nn.Conv2d(ngf*mult, ngf*mult*2, 3, stride=2, padding=1), nn.InstanceNorm2d(ngf*mult*2), nn.ReLU(True)]
        mult = 2**2
        for i in range(n_blocks): model += [ResNetBlock(ngf*mult)]
        for i in range(2):
            mult = 2**(2-i)
            model += [nn.ConvTranspose2d(ngf*mult, int(ngf*mult/2), 3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(int(ngf*mult/2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, 7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)
    def forward(self, input): return self.model(input)

# --- 3. AKILLI VERİ SETİ ---
class SmartDataset(Dataset):
    def __init__(self, image_paths, target_length, transform=None):
        self.image_paths = image_paths
        self.target_length = target_length
        self.transform = transform
    def __len__(self): return self.target_length
    def __getitem__(self, idx):
        img_path = self.image_paths[idx % len(self.image_paths)] # Resim azsa başa döner
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        
        folder_name = os.path.basename(os.path.dirname(img_path))
        base_name = os.path.basename(img_path)
        # İsme index ekliyoruz ki augmentation olursa dosyalar birbirini ezmesin
        unique_filename = f"gan_oulu_idx{idx:05d}_{folder_name}_{base_name}"
        return image, unique_filename

# --- 4. ANA SÜREÇ ---
def generate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Dizin taranıyor...")
    all_source_files = []
    for root, _, files in os.walk(INPUT_DIR):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_source_files.append(os.path.join(root, f))
    
    found_count = len(all_source_files)
    print(f"Kaynak dizinde {found_count} adet resim tespit edildi.")

    if found_count == 0:
        print("HATA: Hiç resim bulunamadı!")
        return

    # Zaten ne kadar üretmişiz?
    existing_count = len([f for f in os.listdir(OUTPUT_DIR) if f.startswith("gan_oulu_")])
    print(f"Halihazırda üretilmiş: {existing_count}")

    if existing_count >= TARGET_NUM_IMAGES:
        print("Hedef 50.000'e zaten ulaşılmış.")
        return

    # EĞER RESİM SAYISI AZSA AUGMENTATION AÇALIM
    if found_count < TARGET_NUM_IMAGES:
        print("Uyarı: Kaynak resim 50k'dan az. Augmentation (Crop/Flip) aktif edildi.")
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20)),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        print("Kaynak resim yeterli. Sadece Resize yapılıyor.")
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    model = ResNetGenerator().to(device)
    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device, weights_only=True)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    model.eval()

    remaining_to_generate = TARGET_NUM_IMAGES - existing_count
    # Dataset'e toplam hedefi değil, kalan hedefi veriyoruz (0'dan başlamasın diye idx kaydıracağız)
    dataset = SmartDataset(all_source_files, target_length=remaining_to_generate, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"\n--- OULU ÜRETİMİ BAŞLADI ---")
    
    with torch.amp.autocast('cuda'), torch.no_grad():
        for i, (batch_images, batch_filenames) in enumerate(dataloader):
            batch_images = batch_images.to(device)
            outputs = model(batch_images)
            
            for j in range(outputs.size(0)):
                # idx'i existing_count ile kaydırıyoruz ki eski dosyalarla çakışmasın
                actual_idx = existing_count + i * BATCH_SIZE + j
                save_name = f"gan_oulu_idx{actual_idx:05d}_{batch_filenames[j].split('_', 3)[-1]}"
                save_image(outputs[j], os.path.join(OUTPUT_DIR, save_name), normalize=True, value_range=(-1, 1))

            sys.stdout.write(f"\rİlerleme: %{((existing_count + (i+1)*BATCH_SIZE)/TARGET_NUM_IMAGES)*100:.2f} | Toplam: {existing_count + (i+1)*BATCH_SIZE}")
            sys.stdout.flush()

    print(f"\n\nBitti! Çıktılar: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate()
