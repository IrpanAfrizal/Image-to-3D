# Image to GLB

Ubah gambar JPG/PNG menjadi model 3D `.glb` berwarna. Semua berjalan lokal di Mac, tanpa API key dan tanpa biaya.

Cara kerja: model **Depth Anything V2 Small** memperkirakan kedalaman gambar, lalu dibuat mesh relief dengan gambar asli sebagai tekstur, sehingga warna GLB sama dengan gambar.

## Menjalankan (macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --port 8000
```

Buka http://127.0.0.1:8000. Pada run pertama, bobot model (sekitar 100 MB) diunduh otomatis. Setelah itu bisa berjalan offline, kecuali font dan viewer 3D di halaman yang dimuat dari CDN.

Disarankan Python 3.11 atau 3.12. Di Apple Silicon, aplikasi otomatis memakai GPU (MPS).

## Lisensi komponen

- Depth Anything V2 **Small**: Apache-2.0 (varian Base/Large/Giant berlisensi non-komersial, jadi jangan diganti tanpa cek lisensi)
- FastAPI, trimesh, transformers: MIT/Apache-2.0
- `<model-viewer>`: Apache-2.0

Tambahkan file `LICENSE` milik Anda sendiri sebelum dipublikasikan ke GitHub.

## Mode Objek 360° (TripoSR, Fase 2)

Mode ini membuat objek 3D utuh dari satu foto memakai TripoSR (MIT), berjalan lokal dan gratis.

```bash
# dari folder image-to-glb, dengan venv aktif
git clone https://github.com/VAST-AI-Research/TripoSR.git
pip install -r requirements.txt
```

Jangan menjalankan `pip install -r TripoSR/requirements.txt`: file itu memasang `torchmcubes` yang sering gagal dibangun di macOS. Aplikasi ini sudah menggantinya dengan scikit-image secara otomatis.

Run pertama mengunduh model TripoSR (sekitar 1,7 GB) dan model penghapus latar rembg (sekitar 170 MB). Hasil terbaik: satu benda dengan latar sederhana. Warna disimpan sebagai vertex color.

Tambahkan `TripoSR/` ke `.gitignore` jika tidak ingin ikut di-commit.
