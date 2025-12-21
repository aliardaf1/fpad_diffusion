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
        # Replay-Attack: Klasör yolunda 'real' ifadesi varsa gerçektir.
        # Örnek Path: .../train/real/client001.mov -> REAL
        # Örnek Path: .../train/attack/fixed/print/client001.mov -> ATTACK
        full_path_str = video_path.lower()
        # Yol ayracına göre bölerek kontrol etmek daha güvenlidir
        path_parts = full_path_str.replace('\\', '/').split('/')
        
        if 'real' in path_parts:
            return 'real'
        else:
            return 'attack'
            
    elif dataset_name == "OULU-NPU":
        # OULU-NPU: Dosya formatı P_S_U_F.avi şeklindedir.
        # Son rakam (File ID): 1 = Real, 2-5 = Attack (Kaynak: OULU Readme.pdf)
        try:
            name_no_ext = filename.rsplit('.', 1)[0]
            parts = name_no_ext.split('_')
            access_type = int(parts[-1]) # Son parça dosya tipidir
            
            if access_type == 1:
                return 'real'
            else:
                return 'attack'
        except Exception as e:
            print(f"[UYARI] OULU Etiket Hatası: {filename} | Hata: {e}")
            return None 

    return None # Bilinmeyen veri seti

def process_video(video_path, dataset_name, subset, label):
    """
    Tek bir videoyu işler: Yüz tespiti -> Kırpma -> Resize -> Kaydetme
    """
    video_name = os.path.basename(video_path).rsplit('.', 1)[0]

    # Kayıt Yolu: data/processed/DatasetAdi/subset/label/
    # Örn: data/processed/ReplayAttack/train/real/
    save_dir = os.path.join(PROCESSED_ROOT, dataset_name, subset, label)
    ensure_dir(save_dir)

    try:
        # OpenCV yerine imageio (ffmpeg backend) kullanımı daha kararlıdır
        reader = imageio.get_reader(video_path, format="ffmpeg")
    except Exception as e:
        print(f"[HATA] Video açılamadı: {video_path} | {e}")
        return

    # Frame sayısını almayı dene, alamazsan sonsuz döngü (enumerate) kullan
    try:
        n_frames = reader.count_frames()
    except:
        n_frames = None

    for frame_idx, frame in enumerate(reader):
        # Sadece belirli aralıklarla kare al (Downsampling)
        if frame_idx % FRAME_INTERVAL != 0:
            continue
        
        # imageio RGB döndürür, RetinaFace RGB bekler. Sorun yok.
        img_rgb = frame

        # Yüz Tespiti (RetinaFace)
        try:
            resp = RetinaFace.detect_faces(img_rgb)
        except Exception:
            resp = {}

        if isinstance(resp, dict) and resp:
            # Birden fazla yüz varsa en büyüğünü (kameraya en yakını) al
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

                # Koordinatları görüntü sınırlarına kırp
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w_img, x2); y2 = min(h_img, y2)

                # Yüzü kes
                face_crop_rgb = frame[y1:y2, x1:x2]

                if face_crop_rgb.size == 0:
                    continue

                try:
                    # 256x256 Boyutlandırma (PSD standardı)
                    face_resized_rgb = cv2.resize(face_crop_rgb, TARGET_SIZE)

                    # Kaydetme: OpenCV BGR beklediği için RGB->BGR çeviriyoruz
                    face_resized_bgr = cv2.cvtColor(face_resized_rgb, cv2.COLOR_RGB2BGR)

                    save_name = f"{video_name}_frame{frame_idx}.jpg"
                    save_path = os.path.join(save_dir, save_name)
                    cv2.imwrite(save_path, face_resized_bgr)
                except Exception:
                    pass
    
    reader.close()

def main():
    # GPU Kontrolü (Bilgi amaçlı)
    print("Mevcut GPU Cihazları:", tf.config.list_physical_devices('GPU'))
    
    # --- VERİ SETİ YAPILANDIRMASI ---
    # Bu yolları kendi bilgisayarınızdaki "raw" veri yollarına göre güncelleyin!
    DATASETS = {
        "ReplayAttack": {
            "root": "./data/raw/ReplayAttack",  
            "subsets": ["train", "devel", "test"], # README.txt'ye göre standart klasörler
            "ext": "*.mov"
        },
        "OULU-NPU": {
            "root": "./data/raw/OULU-NPU",
            "subsets": ["Train_files", "Dev_files", "Test_files"], # OULU standart klasörleri
            "ext": "*.avi" # OULU genellikle .avi veya .mp4'tür. Kod aşağıda ikisine de bakar.
        }
    }

    print(f"İşlem başlıyor... Hedef Çözünürlük: {TARGET_SIZE}")
    print(f"Çıktı Dizini: {PROCESSED_ROOT}\n")

    for dataset_name, config in DATASETS.items():
        raw_root = config["root"]
        if not os.path.exists(raw_root):
            print(f"[UYARI] {dataset_name} kök dizini bulunamadı: {raw_root}")
            print("Lütfen 'DATASETS' sözlüğündeki yolları kontrol edin.")
            continue
            
        print(f"--- {dataset_name} İşleniyor ---")
        
        for subset in config["subsets"]:
            subset_path = os.path.join(raw_root, subset)
            
            # Recursive tarama: Alt klasörlerdeki tüm videoları bulur
            videos = glob.glob(os.path.join(subset_path, "**", config["ext"]), recursive=True)
            
            # Eğer .avi bulamazsa .mp4 dene (OULU için)
            if not videos and dataset_name == "OULU-NPU":
                 videos = glob.glob(os.path.join(subset_path, "**", "*.mp4"), recursive=True)

            print(f"Alt Küme: {subset} | Bulunan Video: {len(videos)}")
            
            # Subset ismini standartlaştır (devel -> dev, Train_files -> train)
            save_subset = subset.lower().replace("_files", "").replace("devel", "dev")

            for video_path in tqdm(videos, desc=f"{dataset_name}/{save_subset}"):
                label = get_label(video_path, dataset_name)
                
                if label:
                    process_video(video_path, dataset_name, save_subset, label)
                else:
                    # Etiket belirlenemezse (örn. bilinmeyen dosya yapısı)
                    pass

    print("\n--- TÜM İŞLEMLER TAMAMLANDI ---")

if __name__ == "__main__":
    main()
