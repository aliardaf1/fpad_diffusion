import os
import cv2
import glob
import imageio.v2 as imageio
from retinaface import RetinaFace
from tqdm import tqdm

# --- AYARLAR ---
RAW_ROOT = "./data/raw/ReplayAttack"
PROCESSED_ROOT = "./data/processed/ReplayAttack"
TARGET_SIZE = (256, 256)  # 320x240 -> 256x256 Upscaling
FRAME_INTERVAL = 5

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_video(video_path, output_dir, subset, label_type):
    # Dosya ismini al
    video_name = os.path.basename(video_path).rsplit('.', 1)[0]

    # Kayıt yolu: processed/ReplayAttack/train/real/videoAdi_frameX.jpg
    save_dir = os.path.join(output_dir, subset, label_type)
    ensure_dir(save_dir)

    # ✅ Video okuma: OpenCV yerine FFmpeg backend (imageio) kullan
    try:
        reader = imageio.get_reader(video_path, format="ffmpeg")
    except Exception as e:
        print(f"[SKIP] Video acilamadi: {video_path} | {e}")
        return

    for frame_idx, frame in enumerate(reader):
        # frame: RGB (imageio genelde RGB verir)
        if frame_idx % FRAME_INTERVAL != 0:
            continue

        # RetinaFace için RGB lazım (frame zaten RGB varsayıyoruz)
        img_rgb = frame

        try:
            resp = RetinaFace.detect_faces(img_rgb)
        except Exception:
            resp = {}

        if isinstance(resp, dict) and resp:
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

                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w_img, x2); y2 = min(h_img, y2)

                face_crop_rgb = frame[y1:y2, x1:x2]

                if face_crop_rgb.size == 0:
                    continue

                try:
                    face_resized_rgb = cv2.resize(face_crop_rgb, TARGET_SIZE)

                    # cv2.imwrite BGR beklediği için RGB -> BGR çevir
                    face_resized_bgr = cv2.cvtColor(face_resized_rgb, cv2.COLOR_RGB2BGR)

                    save_name = f"{video_name}_frame{frame_idx}.jpg"
                    save_path = os.path.join(save_dir, save_name)
                    cv2.imwrite(save_path, face_resized_bgr)
                except Exception:
                    pass

    try:
        reader.close()
    except Exception:
        pass

def main():
    if not os.path.exists(RAW_ROOT):
        print(f"Hata: {RAW_ROOT} bulunamadı.")
        return

    target_subsets = ["replayattack-train", "replayattack-devel", "replayattack-test"]
    print(f"İşlem başlıyor... Hedef Çözünürlük: {TARGET_SIZE}")

    for subset in target_subsets:
        subset_path = os.path.join(RAW_ROOT, subset)
        if not os.path.exists(subset_path):
            print(f"Uyarı: '{subset}' klasörü bulunamadı, geçiliyor.")
            continue

        # mov taraması
        videos = glob.glob(os.path.join(subset_path, "**", "*.mov"), recursive=True)

        print(f"Klasör: {subset} | Video Sayısı: {len(videos)}")

        for video_path in tqdm(videos):
            path_parts = video_path.split(os.sep)
            label = 'real' if 'real' in path_parts else 'attack'
            process_video(video_path, PROCESSED_ROOT, subset, label)

if __name__ == "__main__":
    main()
