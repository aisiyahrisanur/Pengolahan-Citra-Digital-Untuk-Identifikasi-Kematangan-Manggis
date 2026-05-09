**Pengolahan-Citra-Digital-Untuk-Identifikasi-Kematangan-Manggis**

saya menggunakan metode VGG16 dan CNN, sebuah metode yang telah terbukti efektif untuk tugas-tugas klasifikasi gambar dengan arsitektur Utama adalah VGG16 dengan Teknik Transfer Learning
**1. Arsitektur Base Model (VGG16)**
Feature Extraction: Tahap ini mendeteksi pola visual dasar seperti garis, tekstur, hingga bentuk bulat khas buah manggis.
Freezing: Tahap di mana bobot asli VGG16 "dikunci" agar tidak berubah, sehingga model tetap mempertahankan kemampuan deteksi objek yang sudah matang dari ImageNet.
**2. Tahap Adaptasi (Custom Head)**
Flattening: Mengubah hasil ekstraksi fitur yang berbentuk matriks 3D menjadi vektor 1D agar bisa diproses oleh lapisan saraf biasa.
Dense Layer (Fully Connected): Lapisan dengan 256 neuron yang berfungsi mempelajari hubungan antara fitur yang ditemukan dengan kategori kematangan (Mentah, Matang, Busuk).
Dropout (Regularization): Tahapan untuk mencegah model "menghafal" data secara berlebihan (overfitting) dengan cara mematikan sebagian koneksi saraf selama
**3. latihan.Tahap Klasifikasi (Output Layer)**
saya membuat 3 kategori:
  1. Mentah
  2. Matang Sempurna
  3. Busuk/Terlalu Matang**
**4. Tahap Preprocessing Data (ImageDataGenerator)**
model ini dilatih dengan tahapan Augmentasi Data
