import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import random

print("Model yükleniyor...")
model = tf.keras.models.load_model("gtsrb_traffic_sign_model.h5")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess_image(img):
    img_resized = cv2.resize(img, (64, 64))
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

try:
    test_df = pd.read_csv("Test.csv")
except FileNotFoundError:
    print("HATA: Test.csv dosyası bulunamadı! Lütfen dizini kontrol et.")
    exit()

rastgele_ornekler = test_df.sample(n=10)

plt.figure(figsize=(15, 6))

for i, (index, row) in enumerate(rastgele_ornekler.iterrows()):
    img_path = row["Path"] 
    gercek_id = row["ClassId"]
    
    img = cv2.imread(img_path)
    if img is None:
        continue
        
    img_pre = preprocess_image(img)
    img_rgb = cv2.cvtColor(img_pre, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype('float32') / 255.0
    img_batch = np.expand_dims(img_norm, axis=0)
    
    tahminler = model.predict(img_batch, verbose=0)
    tahmin_id = np.argmax(tahminler)
    guven = np.max(tahminler) * 100
    
    plt.subplot(2, 5, i+1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    renk = "green" if tahmin_id == gercek_id else "red"
    baslik = f"Tahmin: {tahmin_id} (%{guven:.0f})\nGerçek: {gercek_id}"
    
    plt.title(baslik, color=renk, fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.suptitle("Gerçek Dünya Test Verisi Tahminleri", fontsize=16, y=1.05)
plt.show()