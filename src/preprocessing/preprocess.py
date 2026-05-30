import os
import cv2
import glob
import imageio.v2 as imageio
from retinaface import RetinaFace
from tqdm import tqdm
import tensorflow as tf

# --- AYARLAR ---
# İşlenmiş verilerin kaydedileceği kök dizin
PROCESSED_ROOT = "./data/processed"

# Hedef görüntü boyutu (PSD Madde 6.1 gereği)
TARGET_SIZE = (256, 256)

# Videodan kaç karede bir örnek alınacağı (Veri boyutunu yönetmek için)
FRAME_INTERVAL = 5

def ensure_dir(path):
    """Klasör yoksa oluşturur."""
    if not os.path.exists(path):
        os.makedirs(path)

def get_label(video_path, dataset_name):
    """
    Veri setine ve dosya ismine/yoluna göre etiketi (real/attack) belirler.
    """
    filename = os.path.basename(video_path)
    
    if dataset_name == "ReplayAttack":
        # Klasör yolunda 'real' klasörü geçiyorsa etiket 'real'dir.
        # Örn: datasets/replayattack/replayattack-train/real/video.mov
        full_path_str = video_path.lower()
        path_parts = full_path_str.replace('\\', '/').split('/')
        
        if 'real' in path_parts:
            return 'real'
        else:
            return 'attack'
            
    elif dataset_name == "OULU-NPU":
        # OULU-NPU: Dosya formatı P_S_U_F.avi
        # Son rakam (File ID): 1 = Real, 2-5 = Attack
        try:
            name_no_ext = filename.rsplit('.', 1)[0]
            parts = name_no_ext.split('_')
            access_type = int(parts[-1]) 
            
            if access_type == 1:
                return 'real'
            else:
                return 'attack'
        except Exception:
            return None 

    return None

def process_video(video_path, dataset_name, subset, label):
    """
    Tek bir videoyu işler: Yüz tespiti -> Kırpma -> Resize -> Kaydetme
    """
    video_name = os.path.basename(video_path).rsplit('.', 1)[0]

    # Kayıt Yolu: data/processed/DatasetAdi/subset/label/
    save_dir = os.path.join(PROCESSED_ROOT, dataset_name, subset, label)
    ensure_dir(save_dir)

    try:
        reader = imageio.get_reader(video_path, format="ffmpeg")
    except Exception as e:
        print(f"[HATA] Video açılamadı: {video_path} | {e}")
        return

    for frame_idx, frame in enumerate(reader):
        if frame_idx % FRAME_INTERVAL != 0:
            continue
        
        img_rgb = frame

        # Yüz Tespiti (RetinaFace)
        try:
            resp = RetinaFace.detect_faces(img_rgb)
        except Exception:
            resp = {}

        if isinstance(resp, dict) and resp:
            max_area = 0
            target_face = None

            for key in resp:
                face = resp[key]
                x1, y1, x2, y2 = face["facial_area"]
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    target_face = (x1, y1, x2, y2)

            if target_face:
                x1, y1, x2, y2 = target_face
                h_img, w_img, _ = frame.shape

                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w_img, x2); y2 = min(h_img, y2)

                face_crop_rgb = frame[y1:y2, x1:x2]

                if face_crop_rgb.size == 0:
                    continue

                try:
                    # 256x256 Boyutlandırma
                    face_resized_rgb = cv2.resize(face_crop_rgb, TARGET_SIZE)
                    # RGB -> BGR (OpenCV için)
                    face_resized_bgr = cv2.cvtColor(face_resized_rgb, cv2.COLOR_RGB2BGR)

                    save_name = f"{video_name}_frame{frame_idx}.jpg"
                    save_path = os.path.join(save_dir, save_name)
                    cv2.imwrite(save_path, face_resized_bgr)
                except Exception:
                    pass
    
    reader.close()

def main():
    # GPU Bilgisi
    print("Mevcut GPU Cihazları:", tf.config.list_physical_devices('GPU'))
    
    # --- VERİ SETİ YAPILANDIRMASI ---
    DATASETS = {
        "ReplayAttack": {
            "root": "./datasets/replayattack",  # Sizin klasör yapınıza göre
            "subsets": ["replayattack-train", "replayattack-devel", "replayattack-test"], 
            "ext": "*.mov"
        },
        "OULU-NPU": {
            "root": "./data/raw/OULU-NPU", # Burayı kendi yolunuza göre kontrol edin
            "subsets": ["Train_files", "Dev_files", "Test_files"], 
            "ext": "*.avi" 
        }
    }

    print(f"İşlem başlıyor... Hedef Çözünürlük: {TARGET_SIZE}")
    print(f"Çıktı Dizini: {PROCESSED_ROOT}\n")

    for dataset_name, config in DATASETS.items():
        raw_root = config["root"]
        
        # Klasör kontrolü
        if not os.path.exists(raw_root):
            print(f"[UYARI] {dataset_name} ana klasörü bulunamadı: {raw_root}")
            print("Lütfen yolun (path) doğru olduğundan ve klasör adının büyük/küçük harf uyumundan emin olun.")
            continue
            
        print(f"--- {dataset_name} İşleniyor ---")
        
        for subset in config["subsets"]:
            subset_path = os.path.join(raw_root, subset)
            
            # Recursive tarama (Alt klasörlerin hepsine bakar)
            videos = glob.glob(os.path.join(subset_path, "**", config["ext"]), recursive=True)
            
            # OULU için alternatif uzantı kontrolü
            if not videos and dataset_name == "OULU-NPU":
                 videos = glob.glob(os.path.join(subset_path, "**", "*.mp4"), recursive=True)

            print(f"Alt Küme: {subset} | Bulunan Video: {len(videos)}")
            
            # --- ÇIKTI KLASÖRÜ DÜZENLEME (TEMİZLİK) ---
            # replayattack-train -> train
            # replayattack-devel -> dev
            # Train_files -> train
            save_subset = subset.lower() \
                .replace("replayattack-", "") \
                .replace("_files", "") \
                .replace("devel", "dev")

            for video_path in tqdm(videos, desc=f"{dataset_name}/{save_subset}"):
                label = get_label(video_path, dataset_name)
                
                if label:
                    process_video(video_path, dataset_name, save_subset, label)

    print("\n--- TÜM İŞLEMLER TAMAMLANDI ---")

if __name__ == "__main__":
    main()