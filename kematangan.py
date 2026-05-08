import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

# Load VGG16 tanpa layer output teratas (include_top=False)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Bekukan (freeze) bobot VGG16 agar tidak berubah saat latihan awal
base_model.trainable = False

#Tambahkan "Kepala" baru untuk klasifikasi manggis
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax') # 3 output: Mentah, Matang, Busuk
])

# Compile Model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

print("Model siap dilatih dengan dataset manggis")

from tensorflow.keras.preprocessing.image import ImageDataGenerator


base_dir = r'C:\Users\risan\Downloads\Semester 3\Cumputer Vision\Dataset_Manggis'

train_datagen = ImageDataGenerator(
    rescale=1./255,            # Normalisasi piksel
    rotation_range=20,         # Augmentasi: putar gambar
    horizontal_flip=True,      # Augmentasi: balik gambar
    validation_split=0.2       # Gunakan 20% data untuk testing otomatis
)

# Load data training
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=10,
    class_mode='categorical',
    subset='training'
)

# Load data validasi
validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=10,
    class_mode='categorical',
    subset='validation'
)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 10,
    epochs=10, # Ulangi proses belajar 10 kali
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 10
)

# Simpan model
model.save('model_manggis_vgg16.h5')
print("✅ Training selesai dan model telah disimpan!")
