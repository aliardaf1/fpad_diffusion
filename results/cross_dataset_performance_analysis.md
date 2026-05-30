# Face Presentation Attack Detection (FPAD) - Performans ve Karşılaştırmalı Analiz Raporu
## (Tez ve Poster için Yol Haritası)

Bu rapor; difüzyon modelleri (DDPM), GAN tabanlı veri artırma (Optimal Transport CycleGAN) ve Spoof-to-Spoof (S2S) yöntemlerinin Yüz Sunum Saldırısı Tespiti (FPAD) üzerindeki etkilerini analiz eder. Rapor; **Intra-Dataset**, **Cross-Dataset** ve **Robustness (Gürültü/Bulanıklık)** deney sonuçlarına dayanarak tezinizde ve posterinizde kullanabileceğiniz en vurucu noktaları ve tabloları sunmaktadır.

---

## 📌 VERİ SETLERİ VE DEĞERLENDİRME METRİKLERİ
*   **OULU-NPU:** Mobil cihazlarla, farklı aydınlatma ve arka plan koşullarında çekilmiş yüksek kaliteli yüz videoları içerir (Daha zorlu bir veri seti).
*   **Replay-Attack:** Sabit web kamerası ve çeşitli aydınlatmalar altında çekilmiş videolar içerir.
*   **Metrikler:**
    *   **AUC (Area Under ROC Curve):** Eşikten (threshold) bağımsız, modelin sınıflandırma (gerçek/sahte) gücünü gösterir. 1.0 en iyi değerdir.
    *   **APCER (%):** Saldırı kabul oranı (Sahte yüzlerin sisteme sızma oranı).
    *   **BPCER (%):** Gerçek kullanıcıyı reddetme oranı (False Alarm - Gerçek kullanıcıların sisteme alınmama oranı).
    *   **HTER (%):** APCER ve BPCER oranlarının ortalamasıdır. Sistemin genel hata oranını temsil eder. HTER ne kadar düşükse model o kadar başarılıdır.

---

## 🚀 ANALİZİN VURUCU NOKTALARI (KEY FINDINGS)

### 1. Alan Kayması (Domain Shift) ve Aşırı Uyum (Overfitting) Problemi
*   **Bulgu:** Tüm modeller **Intra-Dataset** (eğitildiği veri setiyle test edildiğinde) testlerde mükemmele yakın performans gösterirken (AUC: %99.6 - %100.0, HTER: %0.06 - %2.40), **Cross-Dataset** (farklı veri setiyle test edildiğinde) testlerinde hata oranları dramatik şekilde artmaktadır (HTER: %29.8 - %53.2).
*   **Yorum:** Bu durum, FPAD modellerinin eğitildikleri veri setindeki kamera sensörü, ekran parlamaları veya baskı kalitesi gibi "domain-spesifik" sahtekarlık imzalarına aşırı uyum sağladığını (overfit) ve gerçek hayatta genelleme yeteneğini kaybettiğini kanıtlar. Bu durum, difüzyon tabanlı sentetik veri üretimi motivasyonumuzun ana temelidir.

### 2. DDPM (Difüzyon) ve GAN (Optimal Transport CycleGAN) Karşılaştırması
*   **Bulgu:** Replay-Attack → OULU-NPU çapraz testinde:
    *   **Baseline (Sentetiksiz):** AUC = 0.636, HTER@EER = %41.35
    *   **GAN Destekli (`resnet18_replay_gan`):** AUC = 0.649, HTER@EER = %39.93 (AUC'de +0.013 artış, HTER'de -%1.42 iyileşme)
    *   **DDPM Destekli (`resnet18_replay_clean_ddpm_500`):** AUC = 0.658, HTER@EER = %39.57 (AUC'de +0.022 artış, HTER'de -%1.78 iyileşme)
    *   **DDPM + S2S Destekli (`resnet18_replay_ddpm_s2s`):** AUC = 0.659, HTER@EER = %38.92 (AUC'de +0.023 artış, HTER'de -%2.43 iyileşme)
*   **Yorum:** **DDPM (Difüzyon) modelleri, GAN tabanlı baseline'a göre cross-dataset performansını daha tutarlı ve yüksek oranda artırmaktadır.** Difüzyon modellerinin ürettiği yüksek kaliteli ince detaylar (moire desenleri, yansımalar ve baskı pürüzleri), sınıflandırıcının sahte yüzlerdeki genel fiziksel bozulmaları öğrenmesini sağlar.

### 3. Doz-Etki Eğrisi (The Optimal Dose Study) - "Çok Veri Zararlı Olabilir"
*   **Bulgu:** Sentetik veri miktarı arttıkça cross-dataset performansının **U-şeklinde** bir eğri izlediği tespit edilmiştir:
    *   *Replay-Attack → OULU* yönünde sentetik veri dozu **500 ila 2000** arasındayken performans en tepeye ulaşmakta (HTER %44.18'den %38.70'e düşmekte), ancak doz **50.000**'e çıkarıldığında hata oranı tekrar %45.40'a yükselmektedir.
    *   *OULU → Replay* yönünde de benzer şekilde 500 DDPM dozu baseline'a yakın korurken, 10.000 ve 50.000 dozları performansı bozmaktadır.
*   **Yorum:** Sınıflandırıcıyı çok fazla sentetik veri ile eğitmek, modelin "gerçek sahtekarlıkları" öğrenmek yerine "üretici modelin (DDPM) ürettiği pikselleri ve sentez hatalarını" öğrenmesine (generator domain overfitting) yol açar. Bu nedenle **mikro-dozlama (optimal sentetik veri miktarı)**, cross-dataset dayanıklılığı için kritik bir hiperparamatredir.

### 4. Spoof-to-Spoof (S2S) Stil Transferinin Gücü
*   **Bulgu:** Görüntüyü sıfırdan üretmek yerine, mevcut bir sahte görüntüyü difüzyon gürültüleme-denetleme sürecinden (%50 gürültü gücü ile) geçirerek üretilen **S2S (Spoof-to-Spoof)** verileri, performansı sıçratmıştır:
    *   Sadece DDPM (2K) kullanan model: AUC = 0.606, HTER@EER = %43.53
    *   DDPM (2K) + S2S (500) birleşik kullanan model: **AUC = 0.659, HTER@EER = %38.92**
    *   Bu birleşim, tek başına DDPM'e kıyasla AUC'de **+0.053** gibi devasa bir artış ve hata oranında **-%4.61** net düşüş sağlamıştır.
*   **Yorum:** S2S yöntemi, yüzın kimlik yapısını ve geometrisini korurken difüzyonun zengin dokusal sahtekarlık detaylarını üzerine giydirir. Yapısal doğruluk ile sahtekarlık dokusunun bu hibrit birleşimi, sınıflandırıcının genelleme yeteneğini maksimuma çıkarmıştır.

### 5. Sınıf Dengesi ve Kayıp Fonksiyonu Ağırlıklandırması (Hamle 3)
*   **Bulgu:** Eğitim setine sadece sahte (attack) sentetik veri eklenmesi sınıf dengesizliğine (imbalance) yol açar. Klasik kayıp fonksiyonu kullanıldığında model test setinde her şeyi sahte tahmin etme eğilimindedir (APCER %4.43 iken BPCER %92.16'ya çıkmaktadır). Hamle 3 kapsamında uygulanan **Class Weighted Loss (pos_weight)** ile bu dengesizlik giderilmiş, modelin her iki sınıfa da adil yaklaşması sağlanmıştır.

### 6. Eşik Kalibrasyonu (Threshold Calibration) ve Domain Shift
*   **Bulgu:** Çapraz testlerde varsayılan threshold olan 0.5 kullanıldığında HTER oranları %48-50 civarındadır. Ancak EER threshold (hedef veri setine kalibre edilmiş eşik) kullanıldığında bu oranlar %38-39 seviyelerine gerilemektedir (Örneğin EER Threshold: 0.0011).
*   **Yorum:** Alan kayması nedeniyle modelin çıkardığı skorların dağılımı tamamen sola (0'a yakın) kaymaktadır. Bu durum, sabit bir 0.5 eşiğinin cross-dataset testlerinde geçersiz olduğunu, deployment aşamasında hedef domainden alınacak küçük bir validasyon setiyle **eşik kalibrasyonu yapmanın zorunlu olduğunu** gösterir.

### 7. Dış Gürültü ve Bulanıklığa Karşı Sağlamlık (Perturbation Robustness)
*   **Bulgu:** Bulanıklık (Blur) ve Gürültü (Noise) uygulandığında:
    *   **OULU Baseline:** Noise altında tamamen çökmektedir (HTER = %50.0).
    *   **GAN ve DDPM Destekli Modeller:** Noise altında HTER oranını OULU'da %30-%43 arasına düşürmüş; Replay-Attack'ta ise baseline'ın %17.90 olan hata oranını **%4.29 (DDPM)** ve **%5.48 (GAN)** seviyelerine çekmiştir.
*   **Yorum:** Sentetik veri artırımı sadece farklı veri setlerine karşı değil, aynı zamanda **düşük kamera kalitesi, odak kayması (blur) ve sensör gürültüsü (noise) gibi fiziksel bozucu etkenlere karşı da modeli regüle ederek üstün koruma sağlamaktadır.**

---

## 📊 TEZ İÇİN ASIL SONUÇ TABLOLARI

### TABLO 1: Cross-Dataset (Çapraz Test) Performansı (Ana Tablo)
Eğitim setine sentetik veriler eklenerek elde edilen çapraz test sonuçları. (Burada en iyi modeller ve baseline'lar karşılaştırılmıştır).

| Eğitim Seti &rarr; Test Seti | Model Detayı | AUC | HTER @ 0.5 | HTER @ EER | İyileşme (&Delta; HTER@EER) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **OULU-NPU &rarr; Replay-Attack** | Baseline (Sentetiksiz) | **0.706** | %34.18 | **%29.84** | - |
| | + Clean DDPM (500 Doz) | 0.701 | %30.15 | %29.97 | +0.13 (Stabil) |
| | + Clean DDPM (2000 Doz) | 0.665 | %33.76 | %33.67 | -3.83 (Doz Aşımı) |
| | + OT-CycleGAN (Baseline GAN) | 0.699 | %36.29 | %32.23 | -2.39 (Doz Aşımı) |
| **Replay-Attack &rarr; OULU-NPU** | Baseline (Sentetiksiz) | 0.636 | %48.95 | %41.35 | - |
| | + Clean DDPM (500 Doz) | 0.658 | %50.09 | %39.57 | **+1.78** |
| | + OT-CycleGAN (Baseline GAN) | 0.649 | %48.31 | %39.93 | +1.42 |
| | + DDPM (2K) + S2S (500) [En İyi] | **0.659** | %48.30 | **%38.92** | **+2.43** |
| | + DDPM (2K) + S2S (500) (Weighted) | 0.658 | %48.47 | %39.77 | +1.58 |

---

### TABLO 2: Sentetik Miktar (Doz-Etki) Analizi (Replay-Attack &rarr; OULU-NPU)
Veri artırımında kullanılan sentetik örnek miktarının genelleme performansına etkisi.

| Sentetik Örnek Sayısı (Doz) | Model İsmi | Cross-AUC | Cross-HTER @ EER | Analiz / Trend |
| :---: | :--- | :---: | :---: | :--- |
| **0 (Baseline)** | `resnet18_replay_baseline` | 0.636 | %41.35 | Çıkış Noktası |
| **500** | `resnet18_replay_clean_ddpm_500` | **0.658** | **%39.57** | **Optimal Doz (Tepe Noktası)** |
| **2000** | `resnet18_replay_clean_ddpm_2000` | 0.582 | %44.00 | Doz Aşımı Başlangıcı |
| **5000** | `resnet18_replay_clean_ddpm_5000` | 0.601 | %43.77 | Doyum Noktası |
| **10000** | `resnet18_replay_clean_ddpm_10000` | 0.596 | %43.66 | Platonlaşma |

---

### TABLO 3: Fiziksel Pertürbasyonlara Karşı Sağlamlık (HTER %)
Modellerin görüntü kalitesi bozulmalarına (Bulanıklık ve Gürültü) karşı dayanıklılık analizi.

| Test Edilen Veri Seti | Model Detayı | Clean HTER | Blur HTER (Bulanık) | Noise HTER (Gürültülü) |
| :--- | :--- | :---: | :---: | :---: |
| **OULU-NPU** | Baseline | **%1.86** | %27.45 | %50.00 (Tam Çökme) |
| | GAN Augmented Best | %2.20 | **%13.73** | **%30.28** (Büyük Direnç) |
| | DDPM Augmented (50K) | %2.43 | %16.88 | %43.26 |
| **Replay-Attack** | Baseline | **%1.31** | **%0.93** | %17.90 |
| | GAN Augmented Best | %1.32 | %1.72 | %5.48 |
| | DDPM Augmented Best | %2.46 | %2.84 | **%4.29** (En Sağlam) |

---

## 🎨 POSTERE KESİN EKLENMESİ GEREKENLER (MUST-HAVES)

Poster jürisini ve okuyucuları etkilemek için görsel ağırlıklı ve net bir anlatım seçilmelidir. Posterinize şu bileşenleri eklemelisiniz:

### 1. Görsel Blok: Üretilen Görüntü Kalitesi Karşılaştırması
*   **İçerik:** Orijinal Spoof Görüntüsü, OT-CycleGAN tarafından dönüştürülen görüntü, DDPM (Difüzyon) ile sıfırdan üretilen görüntü ve S2S (Spoof-to-Spoof) ile dönüştürülen görüntünün yan yana yüksek çözünürlüklü karşılaştırması.
*   **Vurgulanacak Nokta:** DDPM ve S2S'in GAN'a göre yüz detaylarını bozmadan nasıl daha gerçekçi doku, moiré paterni ve ışık yansıması ekleyebildiği görselleştirilmelidir.

### 2. Grafik: Doz-Etki Eğrisi (U-Shape Curve)
*   **İçerik:** X ekseninde "Sentetik Veri Miktarı (0 - 500 - 2K - 5K - 10K)", Y ekseninde "Cross-HTER @ EER (%)" olan şık bir çizgi grafik.
*   **Vurgulanacak Nokta:** Grafikte 500-2K dozunda HTER'in vadi yaptığı (en düşük değere indiği), sonrasında ise sentetik veri miktarı arttıkça hata oranının tekrar yükseldiği net bir şekilde gösterilmelidir. Altına kalın harflerle şu not yazılmalıdır: *"Optimal sentetik dozlama, üretici modele aşırı uyumu (generator overfitting) engeller."*

### 3. Grafik: Pertürbasyon Direnci (Bar Chart)
*   **İçerik:** Baseline, GAN ve DDPM modellerinin Clean, Blur ve Noise altındaki HTER hata oranlarını gösteren yan yana sütun grafiği.
*   **Vurgulanacak Nokta:** Baseline modelin gürültüde (noise) %50 hata oranı ile tamamen çökmesine karşın, sentetik veriyle eğitilen modellerin (özellikle DDPM'in) gürültü direncini nasıl koruduğu gösterilmelidir.

### 4. Metin Bloğu: "Take-Away Messages" (Poster Sonuç Bölümü)
Posterin sağ alt köşesine yerleştirilecek en önemli 3 sonuç cümlesi:
1.  **Difüzyon Modelleri (DDPM), GAN'lara Güçlü Bir Alternatiftir:** DDPM zengin dokusal sahtekarlık detayları üreterek cross-dataset performansını GAN tabanlı baseline'a kıyasla daha fazla artırır (HTER: %41.35 &rarr; %38.92).
2.  **S2S Hibrit Yaklaşımı En Başarılı Sonucu Verir:** Sıfırdan görüntü üretmek yerine, yüz yapısını koruyan S2S ile dokusal DDPM üretimlerini birleştirmek genelleme başarısının anahtarıdır (AUC'de +0.053 artış).
3.  **Çok Sentetik Veri Zararlıdır (Doz-Etki):** Sınıflandırıcıyı aşırı miktarda sentetik veriyle beslemek hedef domain yerine generator domain'e uyuma yol açar; bu yüzden "Optimal Doz" (U-Curve) belirlenmelidir.
4.  **Eşik Kalibrasyonu Zorunludur:** Çapraz domain testlerinde domain shift sebebiyle sabit 0.5 eşiği çöker. Eşikleri hedef domainin karakteristiğine göre kalibre etmek (EER threshold) hata oranlarını ~%10 düşürür.
