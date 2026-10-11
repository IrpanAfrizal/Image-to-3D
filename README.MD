<div align="center">

# 🧊 Image to GLB

**Ubah gambar JPG/PNG menjadi model 3D `.glb` berwarna — sepenuhnya lokal di macOS, gratis, tanpa API key.**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Platform](<https://img.shields.io/badge/Platform-macOS%20(Apple%20Silicon)-000?logo=apple&logoColor=white>)](#-persyaratan)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#-lisensi)

</div>

---

## ✨ Fitur

- 🖼️ **Dua mode konversi** — _Relief cepat_ dan _Objek 360° (AI)_.
- 🎨 **Warna sesuai gambar asli** — tekstur dipetakan langsung dari JPG/PNG.
- ⚡ **Berjalan lokal** — tanpa API key, tanpa biaya, tanpa koneksi internet setelah setup.
- 🍎 **Native Apple Silicon** — otomatis memakai MPS (Metal Performance Shaders).
- 🧠 **Dua backend AI** untuk mode objek:
  - **SF3D** (Stable Fast 3D) — geometri halus + material PBR
  - **TripoSR** — lebih ringan, cocok untuk GPU kecil
- 🎛️ **UI modern** dengan slider, pratinjau 3D interaktif (`<model-viewer>`), dan progres real-time.
- 📦 **Output satu file** — `.glb` siap dipakai di Blender, Unity, Unreal, Three.js, AR/VR.

---

## 📸 Cara Kerja

```
JPG/PNG ──▶ Preprocessing ──▶ Model AI ──▶ Mesh + Tekstur ──▶ .glb
```

| Mode                     | Model                   | Output                                       |
| ------------------------ | ----------------------- | -------------------------------------------- |
| **Relief cepat**         | Depth Anything V2 Small | Papan relief bertekstur (mesh plane + depth) |
| **Objek 360° (SF3D)**    | Stable Fast 3D          | Mesh utuh + PBR (albedo/metallic/roughness)  |
| **Objek 360° (TripoSR)** | TripoSR                 | Mesh utuh + texture bake 1K–1.5K             |

---

## 🚀 Instalasi

### Persyaratan

- **macOS** (disarankan Apple Silicon M1/M2/M3/M4/M5)
- **Python 3.11 atau 3.12** (3.13 belum stabil untuk semua deps)
- **RAM 16 GB** (8 GB bisa, tapi resolusi objek harus diturunkan)
- **Disk ~5 GB** untuk model (depth 100 MB + TripoSR 1.7 GB + SF3D 1.5 GB)

### Langkah-langkah

```bash
# 1. Clone repositori ini
git clone https://github.com/USERNAME/image-to-glb.git
cd image-to-glb

# 2. Buat virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependensi dasar
pip install -r requirements.txt

# 4. Clone TripoSR (untuk backend objek ringan)
git clone https://github.com/VAST-AI-Research/TripoSR.git

# 5. Install SF3D tanpa menimpa torch/transformers
pip install --no-deps git+https://github.com/Stability-AI/stable-fast-3d.git
```

> **Catatan:** Jangan jalankan `pip install -r TripoSR/requirements.txt`.
> File itu memasang `torchmcubes` yang sering gagal dibangun di macOS.
> Proyek ini sudah menggantinya dengan `scikit-image` secara otomatis.

### Jalankan

```bash
uvicorn main:app --port 8000
```

Buka **[http://127.0.0.1:8000](http://127.0.0.1:8000)** di browser.

> Model akan diunduh otomatis dari Hugging Face pada run pertama
> (~100 MB untuk depth, ~1.7 GB untuk TripoSR, ~1.5 GB untuk SF3D).
> Setelah itu, aplikasi bisa jalan **offline** (kecuali font & viewer 3D dari CDN).

---

## 🎮 Cara Pakai

1. **Jatuhkan** gambar JPG/PNG (maks 15 MB) ke area drop.
2. Pilih mode:
   - **Relief cepat** — untuk papan 3D dengan kedalaman gambar.
   - **Objek 360° (AI)** — untuk model objek utuh dari satu foto.
3. Atur parameter (kedalaman relief / detail mesh / kualitas AI).
4. Klik **Buat model 3D**.
5. Pratinjau 3D muncul → klik **Unduh GLB**.

### Tips hasil terbaik

- ✅ Satu objek jelas dengan latar polos / sederhana
- ✅ Pencahayaan merata, hindari bayangan keras
- ✅ Sudut ¾ depan biasanya paling baik
- ✅ Resolusi gambar ≥ 800 px
- ❌ Hindari objek transparan, reflektif, atau tipis (rambut, kaca, kain)

---

## 🧠 Perbandingan Backend Objek

| Aspek                | **SF3D**                        | **TripoSR**                      |
| -------------------- | ------------------------------- | -------------------------------- |
| Kehalusan geometri   | ⭐⭐⭐⭐⭐                      | ⭐⭐⭐                           |
| Kualitas tekstur     | PBR (albedo+metallic+roughness) | Base color (bake manual)         |
| Kecepatan (M-series) | ~15–25 s                        | ~15–25 s                         |
| Ukuran model         | ~1.5 GB                         | ~1.7 GB                          |
| Lisensi              | Stability Community             | MIT                              |
| Rekomendasi          | **Untuk kualitas terbaik**      | Untuk GPU kecil / kompatibilitas |

> **Kesimpulan:** di Mac 16 GB, gunakan **SF3D** untuk hasil terbaik, **TripoSR** sebagai cadangan.

---

## 📁 Struktur Proyek

```
image-to-glb/
├── main.py                # Backend FastAPI + logika konversi
├── requirements.txt       # Dependensi Python
├── static/
│   └── index.html         # UI (HTML + CSS + JS murni)
├── TripoSR/               # (git-ignored) hasil git clone
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🧪 Troubleshooting

<details>
<summary><b>Error: <code>MPS backend out of memory</code></b></summary>

Turunkan resolusi di UI ke **224 (Cepat)** dan/atau tutup aplikasi lain yang memakai GPU.
Untuk TripoSR, `chunk_size` sudah diturunkan ke 2048 di kode.

</details>

<details>
<summary><b>Error: <code>moderngl</code> gagal create context</b></summary>

Biasanya karena bake texture berjalan di MPS. Aplikasi sudah otomatis pindah ke CPU
saat deteksi MPS. Kalau masih error, jalankan server dengan:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uvicorn main:app --port 8000
```

</details>

<details>
<summary><b>SF3D tidak bisa import</b></summary>

Pastikan install dengan `--no-deps`:

```bash
pip install --no-deps git+https://github.com/Stability-AI/stable-fast-3d.git
python -c "from sf3d.system import SF3D; print('OK')"
```

</details>

<details>
<summary><b>Hasil objek "blocky" atau warna rata</b></summary>

- Naikkan **Kualitas mesh** ke **320 (Halus)**.
- Pastikan memakai **SF3D** (TripoSR cenderung lebih kasar).
- Cek log server: harus muncul `[sf3d] ✅ ...` atau `[triposr] ✅ Texture bake OK`.
</details>

<details>
<summary><b>Hasil jelek dari foto saya</b></summary>

Ini keterbatasan model _single-image reconstruction_ — bukan bug. Sisi belakang objek
selalu "ditebak". Untuk akurasi tinggi, gunakan **multi-view reconstruction** (NeRF/3DGS)
dengan 4–10 foto dari sudut berbeda.

</details>

---

## 🛣️ Roadmap

- [x] Mode relief dengan depth Anything V2
- [x] Mode objek 360° dengan TripoSR
- [x] Integrasi Stable Fast 3D (PBR)
- [x] UI modern dengan pratinjau interaktif
- [ ] Export ke `.obj` dan `.fbx`
- [ ] Batch konversi multi-file
- [ ] Dukungan Linux & Windows (CUDA)
- [ ] Mode multi-view (NeRF/3DGS)

---

## ⚖️ Lisensi Komponen

Proyek ini menggunakan beberapa model dan library pihak ketiga:

| Komponen                           | Lisensi             | Catatan                                   |
| ---------------------------------- | ------------------- | ----------------------------------------- |
| **Depth Anything V2 Small**        | Apache-2.0          | Varian Base/Large/Giant **non-komersial** |
| **Stable Fast 3D (SF3D)**          | Stability Community | Gratis jika pendapatan < $1 juta/tahun    |
| **TripoSR**                        | MIT                 | Bebas dipakai                             |
| **rembg**                          | MIT                 | Model U2Net: Apache-2.0                   |
| **FastAPI, trimesh, transformers** | MIT / Apache-2.0    | —                                         |
| **`<model-viewer>`**               | Apache-2.0          | Dimuat dari CDN                           |

---

## 🤝 Kontribusi

Kontribusi sangat diterima! Silakan:

1. **Fork** repositori ini
2. Buat branch fitur: `git checkout -b fitur-keren`
3. Commit perubahan: `git commit -am 'Menambah fitur keren'`
4. Push: `git push origin fitur-keren`
5. Buka **Pull Request**

Untuk bug besar atau fitur baru, mohon buka **Issue** terlebih dahulu agar bisa didiskusikan.

---

## 🙏 Kredit

- [**Depth Anything V2**](https://github.com/DepthAnything/Depth-Anything-V2) — Lihe Yang et al.
- [**Stable Fast 3D**](https://github.com/Stability-AI/stable-fast-3d) — Stability AI
- [**TripoSR**](https://github.com/VAST-AI-Research/TripoSR) — VAST AI Research & Tripo AI
- [**trimesh**](https://github.com/mikedh/trimesh) — Michael Dawson-Haggerty
- [**rembg**](https://github.com/danielgatis/rembg) — Daniel Gatis

---

## 📜 Lisensi

Kode di repositori ini dilisensikan di bawah **MIT License** — lihat file [LICENSE](LICENSE) untuk detail.

> ⚠️ **Penting:** Model AI pihak ketiga (Depth Anything V2, SF3D, TripoSR) memiliki
> lisensi masing-masing. Jika Anda berencana memakai hasil 3D secara **komersial**,
> periksa dulu lisensi setiap model di tabel di atas.

---

<div align="center">

Dibuat dengan ❤️ di macOS · Kalau proyek ini membantu, beri ⭐ di GitHub!

</div>
