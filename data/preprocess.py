import os
import cv2
import glob
from retinaface import RetinaFace
from tqdm import tqdm

# --- AYARLAR ---
RAW_ROOT = "./data/raw/ReplayAttack"
PROCESSED_ROOT = "./data/processed/ReplayAttack"
TARGET_SIZE = (256, 256) # 320x240 -> 256x256 Upscaling yapılacak
FRAME_INTERVAL = 5 

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_video(video_path, output_dir, subset, label_type):
    # Video okuma ve hata kontrolü
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    # Dosya ismini al
    video_name = os.path.basename(video_path).split('.')[0]
    
    # Kayıt yolu: processed/ReplayAttack/train/real/videoAdi_frameX.jpg
    save_dir = os.path.join(output_dir, subset, label_type)
    ensure_dir(save_dir)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_INTERVAL == 0:
            # RetinaFace için BGR -> RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                # Yüz tespiti (320x240 orijinal görüntüde)
                resp = RetinaFace.detect_faces(img_rgb)
            except:
                resp = {}

            if isinstance(resp, dict):
                # En büyük yüzü bul
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
                    
                    # Sınırları kontrol et
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(w_img, x2); y2 = min(h_img, y2)

                    face_crop = frame[y1:y2, x1:x2]

                    # 256x256 Yeniden Boyutlandırma (Upscaling)
                    try:
                        face_resized = cv2.resize(face_crop, TARGET_SIZE)
                        
                        save_name = f"{video_name}_frame{frame_idx}.jpg"
                        save_path = os.path.join(save_dir, save_name)
                        cv2.imwrite(save_path, face_resized)
                    except:
                        pass

        frame_idx += 1
    cap.release()

def main():
    if not os.path.exists(RAW_ROOT):
        print(f"Hata: {RAW_ROOT} bulunamadı.")
        return

    # SADECE BU 3 KLASÖRÜ İŞLE (Enroll HARİÇ) [cite: 76-78, 93]
    target_subsets = ['train', 'devel', 'test']
    
    print(f"İşlem başlıyor... Hedef Çözünürlük: {TARGET_SIZE}")
    
    for subset in target_subsets:
        subset_path = os.path.join(RAW_ROOT, subset)
        if not os.path.exists(subset_path):
            print(f"Uyarı: '{subset}' klasörü bulunamadı, geçiliyor.")
            continue
            
        # Recursive tarama (attack/fixed, attack/hand, real vb. hepsini bulur)
        # Bu sayede Grandtest protokolüne uymuş oluruz 
        videos = glob.glob(os.path.join(subset_path, "**", "*.mov"), recursive=True)
        
        print(f"Klasör: {subset} | Video Sayısı: {len(videos)}")
        
        for video_path in tqdm(videos):
            # Klasör yolunda 'real' geçiyor mu?
            path_parts = video_path.split(os.sep)
            label = 'real' if 'real' in path_parts else 'attack'
            
            process_video(video_path, PROCESSED_ROOT, subset, label)

if __name__ == "__main__":
    main()