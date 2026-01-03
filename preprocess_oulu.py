import os
import cv2
import glob
from retinaface import RetinaFace
from tqdm import tqdm
import numpy as np

# --- KONFIGÜRASYON ---
# SSD içindeki OULU-NPU ana klasör yolu (Burası sende farklı olabilir, güncellemelisin)
SOURCE_ROOT_DIR = "/media/kullanici/SSD_NAME/OULU-NPU" 

# Proje içindeki hedef klasör
DEST_ROOT_DIR = "./data/processed/OULU-NPU"

# Hedef görüntü boyutu (PSD standardı)
IMG_SIZE = (256, 256)

# Her kaç karede bir kayıt yapılacağı (1: hepsi, 5: her 5 karede bir)
# DDPM için veri çeşitliliği önemli olduğundan çok seyrek yapmamakta fayda var.
FRAME_INTERVAL = 4 

def create_dir_structure():
    """Hedef klasör yapısını oluşturur."""
    subsets = ['Train', 'Dev', 'Test']
    classes = ['Real', 'Spoof']
    
    for subset in subsets:
        for cls in classes:
            path = os.path.join(DEST_ROOT_DIR, subset, cls)
            os.makedirs(path, exist_ok=True)
            print(f"Klasör doğrulandı: {path}")

def get_label_from_filename(filename):
    """
    Dosya ismindeki son numaraya göre etiket belirler.
    Format: Phone_Session_User_File.avi
    File ID: 1 -> Real
    File ID: 2,3,4,5 -> Spoof (Print ve Video Replay)
    """
    base_name = os.path.splitext(os.path.basename(filename))[0]
    try:
        parts = base_name.split('_')
        file_id = int(parts[-1]) # Son parça File ID'dir
        
        if file_id == 1:
            return 'Real'
        else:
            return 'Spoof'
    except:
        return None

def process_video(video_path, subset_name):
    """Videoyu okur, yüzleri bulur ve kaydeder."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    label = get_label_from_filename(video_path)
    if label is None:
        return

    frame_count = 0
    saved_count = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Frame atlama (Sampling)
        if frame_count % FRAME_INTERVAL == 0:
            # RetinaFace ile yüz tespiti
            # Not: RetinaFace yavaştır ama hassastır. Hızlandırmak için 'align=False' denenebilir.
            faces = RetinaFace.detect_faces(frame)
            
            if isinstance(faces, dict): # Yüz bulunduysa
                # En büyük yüzü al (Birden fazla yüz varsa, odaklanılan kişi en büyüğüdür)
                max_area = 0
                target_face = None
                
                for key in faces:
                    identity = faces[key]
                    facial_area = identity["facial_area"] # [x1, y1, x2, y2]
                    width = facial_area[2] - facial_area[0]
                    height = facial_area[3] - facial_area[1]
                    area = width * height
                    
                    if area > max_area:
                        max_area = area
                        target_face = facial_area
                
                if target_face is not None:
                    x1, y1, x2, y2 = target_face
                    
                    # Koordinatların frame dışına taşmamasını sağla
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    
                    face_img = frame[y1:y2, x1:x2]
                    
                    # Yüz görüntüsü geçerliyse resize et ve kaydet
                    if face_img.size > 0:
                        try:
                            # 256x256 Boyutlandırma
                            face_resized = cv2.resize(face_img, IMG_SIZE)
                            
                            save_path = os.path.join(DEST_ROOT_DIR, subset_name, label, f"{video_name}_frame{frame_count}.jpg")
                            
                            # Yüksek kalite JPG olarak kaydet (Disk tasarrufu için PNG yerine JPG %95 kalite)
                            cv2.imwrite(save_path, face_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                            saved_count += 1
                        except Exception as e:
                            pass # Resize hatası olursa geç

        frame_count += 1
    
    cap.release()

def main():
    create_dir_structure()
    
    # Readme'ye göre klasör isimleri: 'Train_files', 'Dev_files', 'Test_files'
    # Kaynak klasördeki yapıya göre bu listeyi eşleştiriyoruz.
    # Genelde indirdiğinizde klasör isimleri şöyledir:
    dataset_parts = {
        'Train_files': 'Train',
        'Dev_files': 'Dev',
        'Test_files': 'Test'
    }

    for source_sub, dest_sub in dataset_parts.items():
        search_path = os.path.join(SOURCE_ROOT_DIR, source_sub, "**", "*.avi")
        # Recursive arama (alt klasörlerde olabilir)
        videos = glob.glob(search_path, recursive=True)
        
        print(f"İşleniyor: {source_sub} -> {dest_sub} | Toplam Video: {len(videos)}")
        
        for video_path in tqdm(videos):
            process_video(video_path, dest_sub)

if __name__ == "__main__":
    # RetinaFace ilk çalışmada ağırlık dosyalarını indirebilir (~100MB)
    main()
