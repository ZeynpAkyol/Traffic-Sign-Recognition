# 🚦 GTSRB - German Traffic Sign Recognition Benchmark (CNN Traffic Sign Classification)

This project is an end-to-end Deep Learning (CNN) project that automatically classifies traffic signs using the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset. The pipeline covers data preprocessing, data balancing, data augmentation, model training, and evaluation steps.

---

## 📂 Project File Structure & Scripts

The project consists of the following core scripts and directories based on your repository structure:

* **`images/`**: Directory containing sample images or dataset assets.
* **`labels.csv`**: CSV file containing label mappings or dataset references.
* **`image_process.py` (Data Preprocessing & Balancing)**:
  * **CLAHE Application:** Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the $L$ channel in the LAB color space to mitigate varying lighting conditions.
  * **Data Augmentation:** Uses OpenCV for random rotations and brightness adjustments to balance underrepresented classes.
  * **Output:** Saves processed data into a compressed `.npz` format (`gtsrb_dataset.npz`).
* **`model.py` (CNN Model Training)**:
  * **Model Architecture:** Implements a Convolutional Neural Network (CNN) with `Conv2D`, `BatchNormalization`, `MaxPooling2D`, `Flatten`, `Dense`, and `Dropout` layers.
  * **Optimization & Callbacks:** Powered by the `Adam` optimizer, `EarlyStopping`, and `ReduceLROnPlateau`.
  * **Output:** Saves the trained model as `gtsrb_traffic_sign_model.h5` and plots performance curves.
* **`prediction_images.py` (Real-World Test Data Evaluation)**:
  * Evaluates the trained model using test data and randomly selected images. Visualizes predictions with Matplotlib (correct predictions in **green**, incorrect in **red**).
* **`prediction_meta.py` (Reference Class Validation)**:
  * Iterates through reference meta images to test whether each class is correctly recognized by the model and outputs the accuracy count.
* **`precition_test.py` (Prediction & Performance Testing)**:
  * Performs additional model prediction tests and validations.

---

## 🛠️ Technologies & Libraries Used

* **Python 3.x**
* **TensorFlow / Keras** (Deep Learning Framework)
* **OpenCV (`cv2`)** (Image Processing & CLAHE)
* **NumPy** (Matrix & Data Operations)
* **Pandas** (CSV Data Management)
* **Scikit-Learn** (Data Splitting & Stratified Split)
* **Matplotlib** (Visualization)
