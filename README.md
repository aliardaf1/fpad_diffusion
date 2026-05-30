# Robust Face Presentation Attack Detection using Diffusion Models

Bu proje, **Yüz Sunum Saldırısı Tespiti (Face Presentation Attack Detection - FPAD)** sistemlerinde sıkça karşılaşılan **alan kayması (domain shift / domain gap)** ve yetersiz genelleme problemini çözmek amacıyla **Gürültü Giderici Difüzyon Olasılık Modelleri (DDPM)** ve **Spoof-to-Spoof (S2S)** stil transferi yöntemlerini araştırmaktadır. Projede difüzyon tabanlı sentetik veri üretimi yöntemleri, geleneksel **Optimal Transport CycleGAN (OT-CycleGAN)** baseline yöntemi ile karşılaştırılmaktadır.

---

## 📌 Proje Özeti
Yüz tanıma ve canlılık algılama sistemleri, eğitildikleri veri setlerinde (Intra-dataset) %99+ gibi yüksek doğruluk oranlarına ulaşmalarına rağmen, farklı bir veri seti veya ortamda (Cross-dataset) test edildiklerinde kamera sensörü, ekran yansımaları, aydınlatma ve baskı kalitesi farklılıkları nedeniyle başarısız olmaktadır. 

Bu çalışmada:
1. **OULU-NPU** ve **Replay-Attack** veri setleri kullanılarak karşılıklı çapraz testler gerçekleştirilmiştir.
2. Sınıflandırıcının (ResNet-18) genelleme yeteneğini artırmak için **DDPM** ile yüksek kaliteli ve çeşitli sahte (spoof) saldırı yüzleri üretilmiştir.
3. Yüz geometrisini koruyarak sadece sahtekarlık dokusu giydiren **S2S (Spoof-to-Spoof)** transfer yöntemi önerilmiş ve test edilmiştir.
4. Sentetik veri miktarının genellemeye etkisini ölçen **Doz-Etki (Dose-Effect) çalışması** yapılmıştır.
5. Sınıf dengesizliğini çözmek için **Class Weighted Loss (Hamle 3)** entegre edilmiştir.
6. Modellerin dış etkenlere (gürültü, bulanıklık) karşı sağlamlığı (Robustness) ölçülmüştür.

---

## 📂 Proje Dizin Yapısı

```text
fpad_diffusion/
├── data/                       # Ham ve işlenmiş veri klasörleri (Gitignore'da ekli)
│   ├── processed/              # MTCNN ile kesilmiş ve hizalanmış yüz verileri
│   ├── synthetic/              # DDPM ve GAN ile üretilmiş sentetik yüz görüntüleri
│   └── S2S/                    # Spoof-to-Spoof (S2S) ile üretilmiş yüz görüntüleri
├── src/                        # Proje kaynak kodları
│   ├── preprocessing/          # Videolardan kare çıkarma ve MTCNN yüz hizalama
│   ├── generation/             # DDPM/DDIM ve S2S veri üretim betikleri
│   ├── classification/         # ResNet-18 model eğitim boru hattı (Baseline & Augmented)
│   ├── evaluation/             # Eşik hesabı, doz çalışması ve sağlamlık testleri
│   └── tests/                  # Donanım ve pipeline test kodları
├── results/                    # Detaylı test sonuçları, HTER/AUC logları ve karşılaştırmalar
├── requirements.txt            # Proje bağımlılıkları listesi
└── README.md                   # Proje dokümantasyonu (Bu dosya)
```

---

## 🚀 Adım Adım Çalıştırma Rehberi

### 1. Önişleme (Preprocessing)
Veri setlerindeki videolardan kareleri ayıklamak ve yüzleri MTCNN ile tespit edip 256x256 piksel boyutunda kırpmak için:
```bash
python src/preprocessing/preprocess_oulu.py
python src/preprocessing/preprocess_replay.py
```

### 2. Sentetik Veri Üretimi (Generation)
*   **DDPM/DDIM ile Sahte Yüz Üretimi:**
    ```bash
    python src/generation/generate_ddim_oulu_v2.py
    python src/generation/generate_ddim_replay_v2.py
    ```
*   **Spoof-to-Spoof (S2S) Üretimi:** Orijinal sahte görüntülere %50 oranında gürültü ekleyip difüzyon modeliyle geri çözerek yüz yapısını bozmadan sahte dokusu üretmek için:
    ```bash
    python src/generation/s2s_generate_fixed.py
    python src/generation/generate_s2s_ddpm.py
    ```

### 3. Sınıflandırıcı Eğitimi (Classification)
ResNet-18 modellerini eğitmek ve sentetik veriyle artırılmış (augmented) konfigürasyonları çalıştırmak için:
*   **OULU-NPU Modelleri:**
    ```bash
    python src/classification/train_clean_oulu_classifiers.py
    ```
*   **Replay-Attack Modelleri (Doz ve S2S Optimizasyonları):**
    ```bash
    python src/evaluation/hamle_1_ve_2.py
    ```
*   **Sınıf Ağırlıklı Kayıp Fonksiyonu (Hamle 3 - Weighted Loss):** Sentetik veri eklemesiyle oluşan sınıf dengesizliğini çözmek için:
    ```bash
    python src/evaluation/hamle_3_weighted.py
    ```

### 4. Değerlendirme ve Sağlamlık Testleri (Evaluation)
*   **Cross-Dataset ve EER Eşik Hesaplama:** Modelleri çapraz veri setlerinde test etmek ve kalibre eşik sonuçlarını almak için:
    ```bash
    python src/evaluation/eer_threshold_eval.py
    ```
*   **Pertürbasyon ve Sağlamlık (Robustness) Testi:** Eğitilen modellerin bulanıklık (Blur) ve gürültü (Noise) altındaki performansını ölçmek için:
    ```bash
    python src/evaluation/robustness_eval.py
    ```

---

## 📈 Ana Bulgular ve Sonuçlar

Detaylı performans analizi ve sonuç tablolarına **[cross_dataset_performance_analysis.md](results/cross_dataset_performance_analysis.md)** dosyasından erişebilirsiniz.

*   **Difüzyon Üstünlüğü:** Replay-Attack &rarr; OULU-NPU çapraz testinde, DDPM destekli veri artırımı GAN tabanlı baseline'a göre daha yüksek genelleme başarısı sunmuştur (HTER: Baseline %41.35 &rarr; DDPM+S2S %38.92).
*   **Doz-Etki Trendi (U-Eğrisi):** Sentetik veri miktarının aşırı artırılması (50.000 adet), sınıflandırıcının üretici modele aşırı uyum sağlamasına (generator domain overfitting) yol açarak performansı bozmaktadır. Optimal dozajlama **500 - 2000** aralığındadır.
*   **S2S Katkısı:** DDPM'e ek olarak Spoof-to-Spoof verilerinin kullanılması AUC skorlarında **+0.053** oranında doğrudan artış sağlamıştır.
*   **Dış Etken Direnci:** Sentetik veri artırımı, modellerin gürültü ve bulanıklık altındaki dayanıklılığını önemli ölçüde artırmıştır (OULU baseline model gürültüde tamamen çökerken, GAN/DDPM modelleri direnç göstermiştir).

---

## 🛠️ Kurulum ve Ortam Kurulumu

### GPU Ortamı Oluşturma ve Aktifleştirme (PowerShell)
```powershell
python -m venv venv_tf_gpu
.\venv_tf_gpu\Scripts\Activate
```

### Gerekli Kütüphanelerin Kurulumu
Sanal ortamı aktifleştirdikten sonra, PyTorch (CUDA 11.8) ve diğer tüm bağımlılıkları tek seferde kurmak için:
```bash
pip install -r requirements.txt
```
*(requirements.txt dosyasında PyTorch için özel CUDA 11.8 index-url'si tanımlanmış olduğu için ekstra bir indirme parametresi belirtmenize gerek yoktur.)*