# 🧠 Brain Tumor Prediction Using MRI Images

## 📌 Project Description

This project predicts the type of brain tumor based on MRI images. Users can upload an MRI scan through a simple web interface, and the app identifies whether the scan indicates:

- Glioma
- Meningioma
- Pituitary tumor
- No tumor

The system also shows:
- 🔍 A confidence percentage for the predicted class
- 📊 A bar graph displaying confidence scores across all classes

The model used is **InceptionV3**, fine-tuned using transfer learning. It is deployed via **Streamlit** to offer instant predictions. This tool is especially helpful for medical professionals to verify their diagnoses and reduce human error.

> ⚙️ Model Type: Supervised Learning  
> 🔬 Frameworks: TensorFlow Keras, Scikit-learn

---

## 🚀 Demo

### 👉 **[Live Web App Link](#)**  

### 📸 **Screenshots of the Web App UI**

<img width="661" height="866" alt="image" src="https://github.com/user-attachments/assets/fa42badb-ce47-4b9c-af28-c6b5dd945a04" />

<img width="720" height="746" alt="image" src="https://github.com/user-attachments/assets/c1d1aeab-b251-4302-8b5d-c01f386fe798" />



---

## 🛠️ Installation Guide

Clone the repository and install the required libraries.

```bash
git clone https://github.com/yourusername/brain-tumor-prediction.git
cd brain-tumor-prediction
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧠 Model Details

### Model: **InceptionV3 (Transfer Learning)**
The model was trained using augmented and normalized images. It was fine-tuned on a balanced dataset across the four classes.

### ✅ **Evaluation Metrics**

- Accuracy: 0.8740
- Precision: 0.8763
- Recall: 0.8740
- F1 Score: 0.8738

### 📊 **Classification Report**

<img width="608" height="318" alt="image" src="https://github.com/user-attachments/assets/e876a379-64fc-4e22-89aa-89e1b32eb275" />

### 🧪 Confusion Matrix

<img width="493" height="463" alt="image" src="https://github.com/user-attachments/assets/6f225fe7-b52f-49e8-8b0f-c36278d1705f" />

---

## 📈 Results

- The InceptionV3 model achieved an accuracy of 87.4% on the test set.
- The model performs best on glioma and no tumor cases, with slight misclassification between pituitary and meningioma classes.
- Data preprocessing steps included:
-- Resizing images
-- Normalization
-- Data augmentation (rotation, zoom, flip, etc.)

---

## 👨‍⚕️ Why This Matters
- Supports medical professionals in diagnosing brain tumors.
- Reduces chances of misdiagnosis through AI assistance.
- Can be a useful second opinion tool in clinical practice.

---

## 📬 Contact

For suggestions or contributions, feel free to open an issue or pull request.
