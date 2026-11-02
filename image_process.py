import os
import cv2
import numpy as np
import random
import shutil
from tqdm import tqdm


RAW_INPUT_DIR = "Train"               
BALANCED_DIR = "Balanced_Train"      
OUTPUT_FILE = "gtsrb_dataset.npz"     
TARGET_COUNT = 2500                   
IMAGE_SIZE = (64, 64)                 


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess_image(img):
    img_resized = cv2.resize(img, IMAGE_SIZE)
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def augment_image(img):
    rows, cols, _ = img.shape
    
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    img_aug = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

    hsv = cv2.cvtColor(img_aug, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    brightness_factor = random.uniform(0.7, 1.3)
    v = np.clip(v * brightness_factor, 0, 255).astype(np.uint8)
    hsv = cv2.merge((h, s, v))
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def adim1_veriyi_dengele():
    if os.path.exists(BALANCED_DIR):
        shutil.rmtree(BALANCED_DIR)
    os.makedirs(BALANCED_DIR)

    for class_id in range(43):
        class_input_path = os.path.join(RAW_INPUT_DIR, str(class_id))
        class_output_path = os.path.join(BALANCED_DIR, str(class_id))
        os.makedirs(class_output_path, exist_ok=True)
        
        if not os.path.exists(class_input_path):
            continue
            
        image_files = [f for f in os.listdir(class_input_path) if f.endswith('.png') or f.endswith('.jpg')]
        current_count = len(image_files)
        
        processed_images = [] 
        
        for img_name in image_files:
            img_path = os.path.join(class_input_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                preprocessed_img = preprocess_image(img)
                processed_images.append(preprocessed_img)
                save_path = os.path.join(class_output_path, f"orig_{img_name}")
                cv2.imwrite(save_path, preprocessed_img)
                
        deficit = TARGET_COUNT - current_count
        if deficit > 0 and len(processed_images) > 0:
            for i in range(deficit):
                base_img = random.choice(processed_images)
                aug_img = augment_image(base_img)
                save_path = os.path.join(class_output_path, f"aug_{i}.png")
                cv2.imwrite(save_path, aug_img)
                
        print(f"Sınıf {class_id}: Orijinal ({current_count}) + Sentetik ({max(0, deficit)}) = Toplam {TARGET_COUNT}")

def adim2_numpy_yap():
    X = []
    y = []
    
    for class_id in tqdm(range(43), desc="Sınıflar işleniyor"):
        class_path = os.path.join(BALANCED_DIR, str(class_id))
        if not os.path.exists(class_path):
            continue
            
        for img_name in os.listdir(class_path):
            if img_name.endswith('.png') or img_name.endswith('.jpg'):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    X.append(img_rgb)
                    y.append(class_id)
                    
    X = np.array(X, dtype=np.uint8) 
    y = np.array(y, dtype=np.int32)
    
    print(f"\nVeri boyutları:")
    print(f"Görseller (X): {X.shape}")
    print(f"Etiketler (y): {y.shape}")
    
    np.savez_compressed(OUTPUT_FILE, images=X, labels=y)

if __name__ == "__main__":
    adim1_veriyi_dengele()
    adim2_numpy_yap()