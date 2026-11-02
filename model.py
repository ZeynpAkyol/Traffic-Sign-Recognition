import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


data = np.load("gtsrb_dataset.npz")

X_ham = data['images']
y_ham = data['labels']

print(f"Toplam {len(X_ham)} adet görsel başarıyla yüklendi.")

X_train, X_val, y_train, y_val = train_test_split(
    X_ham, y_ham, test_size=0.2, random_state=42, stratify=y_ham
)

print(f"Eğitim verisi boyutu: {X_train.shape}")
print(f"Doğrulama verisi boyutu: {X_val.shape}\n")


X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0


plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[i])
    plt.title(f"Sınıf: {y_train[i]}")
    plt.axis('off')
plt.suptitle("Eğitim Setinden Örnek Görseller", fontsize=16)
plt.tight_layout()
plt.show()


print("\nModel oluşturuluyor...")
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5), 
    layers.Dense(43, activation='softmax') 
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=3, 
    min_lr=0.00001,
    verbose=1
)

EPOCHS = 30
BATCH_SIZE = 64

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr]
)

print("\nEğitim tamamlandı! Model kaydediliyor...")
model.save("gtsrb_traffic_sign_model.h5")
print("Model 'gtsrb_traffic_sign_model.h5' adıyla başarıyla kaydedildi.")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Eğitim Kaybı (Train Loss)')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı (Val Loss)')
plt.title('Model Kaybı (Loss)')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Eğitim Başarısı (Train Acc)')
plt.plot(history.history['val_accuracy'], label='Doğrulama Başarısı (Val Acc)')
plt.title('Model Başarısı (Accuracy)')
plt.xlabel('Epoch')
plt.ylabel('Başarı')
plt.legend()

plt.tight_layout()
plt.show()