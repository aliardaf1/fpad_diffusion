import os, random, shutil

src = "/home/undergrad25_1/Desktop/fpad_diffusion/data/synthetic/OULU-NPU/DDPM_spoof/spoof"
dst = "/home/undergrad25_1/Desktop/mode_collapse_test"
os.makedirs(dst, exist_ok=True)

all_files = os.listdir(src)
random.seed(123)  # tekrar üretilebilir olsun
samples = random.sample(all_files, 30)
for f in samples:
    shutil.copy(os.path.join(src, f), dst)

print(f"30 random örnek kopyalandı: {dst}")
