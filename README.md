# Robust Face Presentation Attack Detection using Diffusion Models

Bu proje, Yüz Sunum Saldırısı Tespiti (FPAD) sistemlerinin dayanıklılığını artırmak için Difüzyon Modelleri (DDPM) tabanlı sentetik veri üretimi yöntemini araştırmaktadır.

## Proje Hedefleri
1. Baseline (OT-CycleGAN) modelinin yeniden üretimi.
2. DDPM ile yüksek kaliteli ve çeşitli spoof saldırı verileri üretimi.
3. Cross-dataset (Replay-Attack -> OULU-NPU) genelleştirme performansının artırılması.

## Kurulum
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Veri Setleri
Projede Replay-Attack ve OULU-NPU veri setleri kullanılmaktadır.

## GPU Ortamı oluşturmak için
python -m venv venv_tf_gpu
## Aktive etmek için PowerSHellden 
.\venv_tf_gpu\Scripts\Activate