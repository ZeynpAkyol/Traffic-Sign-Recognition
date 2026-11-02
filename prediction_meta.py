import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("gtsrb_traffic_sign_model.h5")
print("Model başarıyla yüklendi!\n")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess_image(img):
    img_resized = cv2.resize(img, (64, 64))
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def tabela_tahmin_et(gorsel_yolu):
    img = cv2.imread(gorsel_yolu)
    
    if img is None:
        print(f"Hata: {gorsel_yolu} bulunamadı veya okunamadı!")
        return None
        
    img_preprocessed = preprocess_image(img)
    
    img_rgb = cv2.cvtColor(img_preprocessed, cv2.COLOR_BGR2RGB)
    
    img_normalized = img_rgb.astype('float32') / 255.0

    img_batch = np.expand_dims(img_normalized, axis=0)

    tahminler = model.predict(img_batch, verbose=0)
    tahmin_edilen_sinif = np.argmax(tahminler)
    guven_skoru = np.max(tahminler) * 100
    
    print(f"Görsel: {gorsel_yolu:<25} | Tahmin ID: {tahmin_edilen_sinif:02d} | Güven: %{guven_skoru:.2f}")
    return tahmin_edilen_sinif

print("Tahminler Başlıyor...\n" + "-"*60)

count=0
for i in range(43):
    test_gorseli = f"Meta/{i}.png" 
    if i==tabela_tahmin_et(test_gorseli):
        count+=1

print(count)