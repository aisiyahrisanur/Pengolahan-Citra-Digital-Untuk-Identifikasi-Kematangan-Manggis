import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Load model 
model = load_model('model_manggis_vgg16.h5')

def deteksi_kematangan_manggis(img_path):
    #Load dan Preprocess gambar tes
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalisasi harus sama dengan saat training

    #Prediksi
    predictions = model.predict(img_array)
    
    # Ambil index dengan probabilitas tertinggi
    # Urutan class biasanya alfabetis sesuai folder: ['busuk', 'matang', 'mentah']
    # Cek urutan aslinya dengan train_generator.class_indices di file training
    class_names = ['Busuk/Terlalu Matang', 'Matang Sempurna', 'Mentah']
    
    result_index = np.argmax(predictions)
    label = class_names[result_index]
    confidence = predictions[0][result_index] * 100

    # Tampilkan Hasil
    plt.imshow(img)
    plt.title(f"Prediksi: {label} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()

    print(f"\nHasil Analisis: {label}")
    print(f"Tingkat Kepercayaan: {confidence:.2f}%")

# Panggil fungsi
deteksi_kematangan_manggis('manggis.jpg')