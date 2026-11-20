# User Drowsiness Detection System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh-orange.svg)](https://developers.google.com/mediapipe)
[![Pygame](https://img.shields.io/badge/Pygame-Mixer%20Audio-red.svg)](https://www.pygame.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Sistem Computer Vision waktu nyata (*real-time*) untuk mendeteksi rasa kantuk (*drowsiness*) dan kelelahan (*fatigue/yawning*) pada pengemudi kendaraan maupun pengguna workstation. Dibangun dengan standar *clean code*, modular, memiliki penanganan error yang tangguh, serta audio engine *non-blocking*.

---

## 📋 Daftar Isi
1. [Fitur Utama](#-fitur-utama)
2. [Arsitektur & Landasan Matematis](#-arsitektur--landasan-matematis)
3. [Struktur Proyek](#-struktur-proyek)
4. [Instalasi & Persiapan](#-instalasi--persiapan)
5. [Cara Penggunaan](#-cara-penggunaan)
6. [Opsi Argumen CLI](#-opsi-argumen-cli)
7. [Kontrol Keyboard (Runtime Hotkeys)](#-kontrol-keyboard-runtime-hotkeys)
8. [Panduan Kalibrasi Sensitivitas](#-panduan-kalibrasi-sensitivitas)
9. [Audio Alert & Fallback Engine](#-audio-alert--fallback-engine)

---

## 🌟 Fitur Utama

- **MediaPipe Face Mesh (468 Titik Landmark):** Pelacakan kontur kelopak mata dan bibir berpresisi tinggi dengan opsi `refine_landmarks=True`.
- **Kalkulasi EAR (Eye Aspect Ratio):** Formula 6-titik per mata berbasis jarak Euclidean untuk mendeteksi penutupan mata.
- **Deteksi Menguap (Mouth Aspect Ratio - MAR):** Mendeteksi indikasi awal kelelahan sebelum pengemudi mengalami *microsleep*.
- **Non-blocking Audio Alert (Pygame Mixer):** Alarm berbunyi secara asinkron tanpa menyebabkan frame *freeze* atau *stuttering* pada tampilan OpenCV.
- **NumPy Fallback Buzzer Synthesizer:** Jika file `alarm.wav` tidak ada, sistem otomatis meracik nada sinus ganda darurat (880 Hz & 1100 Hz) langsung ke audio buffer.
- **Interactive Auto-Calibration:** Kalibrasi profil mata otomatis selama 3 detik untuk menyesuaikan bentuk mata setiap individu.
- **Heads-Up Display (HUD) & Visual Alarms:** Tampilan visual semi-transparan, nilai live EAR/MAR, FPS counter, status badge berwarna, dan efek *flashing border* merah ketika kantuk terdeteksi.
- **Dukungan Input Fleksibel:** Kompatibel dengan *live webcam* maupun file rekaman dashcam (`.mp4`, `.avi`).

---

## 📐 Arsitektur & Landasan Matematis

### 1. Eye Aspect Ratio (EAR)
EAR dihitung untuk setiap mata (kiri dan kanan) menggunakan jarak Euclidean 2 dimensi antar landmark kelopak mata:

$$\text{EAR} = \frac{||\mathbf{p}_2 - \mathbf{p}_6|| + ||\mathbf{p}_3 - \mathbf{p}_5||}{2 \cdot ||\mathbf{p}_1 - \mathbf{p}_4||}$$

$$\text{EAR}_{\text{avg}} = \frac{\text{EAR}_{\text{left}} + \text{EAR}_{\text{right}}}{2}$$

#### Pemetaan Titik Landmark MediaPipe:
- **Mata Kiri:**
  - Sudut Horizontal: $\mathbf{p}_1 = 33$, $\mathbf{p}_4 = 133$
  - Pasangan Vertikal 1: $\mathbf{p}_2 = 160$, $\mathbf{p}_6 = 144$
  - Pasangan Vertikal 2: $\mathbf{p}_3 = 158$, $\mathbf{p}_5 = 153$
- **Mata Kanan:**
  - Sudut Horizontal: $\mathbf{p}_1 = 362$, $\mathbf{p}_4 = 263$
  - Pasangan Vertikal 1: $\mathbf{p}_2 = 385$, $\mathbf{p}_6 = 380$
  - Pasangan Vertikal 2: $\mathbf{p}_3 = 387$, $\mathbf{p}_5 = 373$

---

### 2. Mouth Aspect Ratio (MAR) untuk Menguap
Untuk mendeteksi mulut yang menganga saat menguap:

$$\text{MAR} = \frac{||\mathbf{p}_{13} - \mathbf{p}_{14}|| + ||\mathbf{p}_{82} - \mathbf{p}_{87}|| + ||\mathbf{p}_{312} - \mathbf{p}_{317}||}{2 \cdot ||\mathbf{p}_{78} - \mathbf{p}_{308}||}$$

---

### 3. State Machine Logika Deteksi
```
                      [ Wajah Terdeteksi ]
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
       EAR ≥ Threshold                 EAR < Threshold
       (Mata Terbuka)                  (Mata Terpejam)
               │                               │
       Reset Timer Kantuk              Mulai Timer Terpejam
               │                               │
        STATUS: FOCUSED              Waktu Terpejam ≥ 2.0s?
               │                        ├── Ya  ──► STATUS: DROWSY (ALARM BUNYI)
        Alarm Hening                    └── Tidak ─► Status Masih Normal (Berkedip biasa)
```

---

## 📁 Struktur Proyek

```plaintext
projek2/
├── alarm.wav            # File audio alarm default (dapat diganti)
├── main.py              # Script utama (modular & production-ready)
├── README.md            # Dokumentasi lengkap sistem
└── requirements.txt     # Daftar dependensi versi terkunci
```

---

## 💻 Instalasi & Persiapan

### 1. Prasyarat Sistem
- Python versi 3.10 atau lebih baru.
- Webcam bawaan / USB Webcam terhubung.
- Output speaker / headset aktif untuk alarm audio.

### 2. Buat & Aktifkan Virtual Environment (Direkomendasikan)
```bash
# Windows (PowerShell / CMD)
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

---

## 🚀 Cara Penggunaan

### 1. Menjalankan Langsung dengan Webcam Default (Index 0)
```bash
python main.py
```

### 2. Menjalankan dengan File Video (Offline / Dashcam Footage)
```bash
python main.py --video sample_driver.mp4
```

### 3. Mengatur Threshold Kustom via Terminal
```bash
python main.py --camera 0 --ear-threshold 0.20 --duration 2.5
```

---

## ⚙️ Opsi Argumen CLI

| Argumen | Tipe | Default | Deskripsi |
|---|---|---|---|
| `--camera` | `int` | `0` | Nomor index perangkat kamera / webcam. |
| `--video` | `str` | `None` | Path ke file rekaman video pengganti webcam. |
| `--ear-threshold` | `float` | `0.22` | Ambang batas nilai EAR mata tertutup. |
| `--duration` | `float` | `2.0` | Durasi mata terpejam berturut-turut (detik) pemicu alarm. |
| `--alarm-audio` | `str` | `"alarm.wav"` | Path ke file audio suara alarm kustom. |
| `--no-yawn` | `flag` | `False` | Menonaktifkan deteksi kelelahan akibat menguap (MAR). |

---

## 🎮 Kontrol Keyboard (Runtime Hotkeys)

Saat jendela aplikasi sedang aktif, Anda dapat berinteraksi menggunakan tombol berikut:

| Tombol | Fungsi |
|:---:|---|
| **`Q`** / **`ESC`** | **Graceful Shutdown**: Menghentikan audio alarm, melepas kamera, dan menutup aplikasi secara aman. |
| **`C`** | **Auto-Calibration**: Melakukan kalibrasi baseline mata terbuka selama 3 detik untuk menyesuaikan threshold otomatis. |
| **`+`** / **`=`** | **Menaikkan Sensitivitas**: Menambah `EAR_THRESHOLD` sebesar `+0.01`. |
| **`-`** / **`_`** | **Menurunkan Sensitivitas**: Mengurangi `EAR_THRESHOLD` sebesar `-0.01`. |
| **`M`** | **Toggle Deteksi Menguap**: Mengaktifkan atau menonaktifkan fitur pendeteksi mulut menguap (MAR). |

---

## 🎯 Panduan Kalibrasi Sensitivitas

Setiap orang memiliki karakteristik mata yang berbeda (misalnya pemakai kacamata, bentuk mata sipit/lebar, atau kondisi pencahayaan kabin).

### Metode 1: Fitur Auto-Calibration Otomatis (Sangat Dianjurkan)
1. Duduk dengan posisi berkendara normal dan tatap layar dengan mata terbuka rileks.
2. Tekan tombol **`C`** pada keyboard.
3. Tunggu selama **3 detik** hingga progres mencapai 100%.
4. Sistem otomatis mengukur baseline mata terbuka Anda dan menetapkan threshold baru pada **72%** dari baseline natural tersebut.

### Metode 2: Kalibrasi Manual
Perhatikan nilai `EAR Avg` pada kartu HUD pojok kiri atas:
- **Mata Terbuka Normal:** Biasanya berkisar antara `0.28 - 0.36`.
- **Mata Terpejam Rapat:** Biasanya berkisar antara `0.08 - 0.16`.

**Aturan Penyesuaian:**
- Jika alarm sering berbunyi padahal hanya berkedip biasa (*False Alarm*), tekan tombol **`-`** untuk menurunkan threshold (misal ke `0.19 - 0.20`).
- Jika mata sudah terpejam tetapi sistem lambat mendeteksi, tekan tombol **`+`** untuk menaikkan threshold (misal ke `0.23 - 0.25`).

---

## 🔊 Audio Alert & Fallback Engine

Modul [`AudioAlertEngine`](file:///c:/shidqi_lab/projek2/main.py#L182-L289) bekerja dengan prinsip *fail-safe*:
1. **File Audio Lokal:** Sistem akan mencoba memuat file `alarm.wav` secara default.
2. **Fallback Synthesizer Mandiri:** Jika file `alarm.wav` terhapus atau tidak ditemukan, modul secara otomatis membuat gelombang suara sintetis menggunakan array sinusoidal **NumPy** (frekuensi darurat 880 Hz dan 1100 Hz berdenyut) langsung ke buffer audio **Pygame**. Sistem dipastikan **tidak akan pernah crash** karena ketiadaan file suara.
3. **Threading Asinkron:** Pemutaran audio dilakukan pada channel mixer terisolasi sehingga perulangan frame kamera tetap berjalan lancar pada 30-60 FPS tanpa jeda (*non-blocking*).

---

## 📄 Lisensi
Didistribusikan di bawah lisensi **MIT**. Bebas digunakan, dimodifikasi, dan diintegrasikan ke proyek komersial maupun riset akademik.
