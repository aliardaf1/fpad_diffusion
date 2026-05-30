import os
import cv2
import glob
import torch
from tqdm import tqdm
import logging
from facenet_pytorch import MTCNN

# --- LOGGER YAPILANDIRMASI ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- DONANIM VE AYARLAR ---
# PyTorch cihaz yönetimi
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.info(f"Kullanılan Cihaz: {device}")

TARGET_SIZE = (256, 256)
FRAME_INTERVAL = 5 

# --- GÜNCEL YOLLAR ---
DATASET_ROOT = "/media/undergrad25_1/Data/oulu"
OUTPUT_ROOT = "/home/undergrad25_1/Desktop/fpad_diffusion/data/processed/OULU-NPU"

# --- MTCNN MODELİ BAŞLATMA ---
# keep_all=True: Birden fazla yüzü algıla
# post_process=False: Görüntüyü normalize etmeden ham koordinatları al
detector = MTCNN(keep_all=True, device=device, post_process=False)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def get_oulu_metadata(video_path):
    """
    OULU-NPU: P_S_U_F.avi
    F=1 -> Real, F=2..5 -> Attack
    """
    filename = os.path.basename(video_path)
    try:
        f_type = int(filename.rsplit('.', 1)[0].split('_')[-1])
        label = "real" if f_type == 1 else "attack"
        sub_label = f"type_{f_type}" if f_type > 1 else "real"
        return label, sub_label
    except Exception:
        return None, None

def process_video(video_path, subset_name):
    label, sub_label = get_oulu_metadata(video_path)
    if not label: return

    video_name = os.path.basename(video_path).rsplit('.', 1)[0]
    save_dir = os.path.join(OUTPUT_ROOT, subset_name, label, sub_label, video_name)
    ensure_dir(save_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Video açılamadı: {video_name}")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
            
        if frame_idx % FRAME_INTERVAL == 0:
            # MTCNN RGB formatında çalışır
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                # Yüz tespiti (boxes: [x1, y1, x2, y2])
                boxes, _ = detector.detect(img_rgb)
                
                if boxes is not None and len(boxes) > 0:
                    # En büyük yüzü seçme mantığı (Alana göre)
                    # boxes[:, 2] - boxes[:, 0] -> Genişlik
                    # boxes[:, 3] - boxes[:, 1] -> Yükseklik
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    best_box_idx = areas.argmax()
                    box = boxes[best_box_idx]
                    
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Sınır kontrolleri
                    h, w, _ = frame.shape
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                    face_crop = frame[y1:y2, x1:x2]
                    
                    if face_crop.size > 0:
                        resized = cv2.resize(face_crop, TARGET_SIZE)
                        save_path = os.path.join(save_dir, f"{video_name}_f{frame_idx}.jpg")
                        cv2.imwrite(save_path, resized)
            except Exception as e:
                logging.warning(f"Hata ({video_name} - Frame {frame_idx}): {e}")
        
        frame_idx += 1

    cap.release()

def main():
    subset = "Train_files"
    subset_path = os.path.join(DATASET_ROOT, subset)
    
    if not os.path.exists(subset_path):
        logging.error(f"Dizin bulunamadı: {subset_path}")
        return

    videos = glob.glob(os.path.join(subset_path, "*.avi"))
    logging.info(f"İşlem başlıyor. Bulunan video sayısı: {len(videos)}")

    for v_path in tqdm(videos, desc="OULU Train İşleniyor (PyTorch)"):
        process_video(v_path, "train")

if __name__ == "__main__":
    main()
