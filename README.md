# 🎯 Crowd Emotion & Violence Detection using CNN and VGG16

This project is a **real-time AI-powered CCTV monitoring system** designed to detect **human emotions** from faces and **violent activities** in live video streams using **Deep Learning models (CNN & VGG16)**.  
It enables enhanced **public safety, surveillance**, and **crowd behavior analysis** by intelligently analyzing visual data from cameras in real time.

---

## 🌟 Key Highlights

- 🔹 Real-time face detection using **Haar Cascade Classifier**  
- 🔹 Emotion recognition via **VGG16 with Transfer Learning**  
- 🔹 Violence detection using a **custom-built CNN model**  
- 🔹 Supports input from:
  - 💻 Webcam  
  - 📹 CCTV / IP Cameras (RTSP stream)  
- 🔹 Provides:
  - Face bounding box visualization  
  - Emotion detection with confidence score  
  - Violence probability alerts with threshold indicators  

---

## 🧠 Technology Stack

| Category | Tools / Frameworks |
|-----------|--------------------|
| **Language** | Python 3.11 |
| **Deep Learning** | TensorFlow, Keras |
| **Computer Vision** | OpenCV |
| **Feature Extraction** | VGG16 Pretrained Model |
| **Face Detection** | Haar Cascade Classifier |
| **Frontend / Demo** | Streamlit |
| **Utilities** | NumPy, Matplotlib, JSON |

---

## 📂 Project Directory Structure

crowd_emotion_violence/
│
├── data/ # Training dataset
│
├── haarcascades/ # Face detection XML files
│ └── haarcascade_frontalface_default.xml
│
├── models/ # Pre-trained and custom models
│ ├── emotion_model_vgg16.h5
│ ├── violence_model_cnn.h5
│ └── emotion_classes.json
│
├── run_realtime_cctv.py # Real-time detection via OpenCV
├── train_emotion_vgg16.py # Emotion model training script
├── train_violence_cnn.py # Violence model training script
├── app.py # Streamlit demo web app
├── requirements.txt # Python dependencies
├── .gitignore
└── README.md

text

---

## ⚙️ Installation & Setup Guide

### 1️⃣ Clone the Repository
git clone https://github.com/NAUFFAL-2004/crowd-emotion-violence-detection.git
cd crowd-emotion-violence-detection

text

### 2️⃣ Create a Virtual Environment
py -3.11 -m venv venv
.\venv\Scripts\activate

text

### 3️⃣ Install Dependencies
pip install -r requirements.txt

text

### 4️⃣ Run Real-Time Desktop Detector
python run_realtime_cctv.py

text
➡️ Press **Q** anytime to stop live detection.

### 5️⃣ Launch Web Interface (Streamlit)
streamlit run app.py

text
🌐 Open your browser and visit:  
[**http://localhost:8501**](http://localhost:8501)

---

## 📸 Example Output

Include a representative screenshot after uploading it to GitHub:  

<img width="805" height="636" alt="Screenshot 2025-12-07 094210" src="https://github.com/user-attachments/assets/6f4a2199-406c-4921-aa3c-46a55617ce51" />


---

## 📊 Model Information

### 🧩 Emotion Detection Model
- **Architecture:** VGG16 + Custom Dense Layers  
- **Input Shape:** 224 × 224 × 3  
- **Output Classes:**  
  - Angry  
  - Happy  
  - Neutral  
  - Sad  
  - Scared  

### 🔥 Violence Detection Model
- **Architecture:** Custom CNN  
- **Input Shape:** 128 × 128 × 3  
- **Classification Output:**  
  - `0` → Non-Violent  
  - `1` → Violent  

---

## 💡 Core Applications

- 🏙️ Smart city surveillance  
- 🛡️ Public safety and crowd monitoring  
- ✈️ Airport / railway station security  
- 🏫 School and campus surveillance  
- 🎭 Crowd behavior analytics  

---

## 🚀 Future Enhancements

- ⚡ GPU acceleration for faster real-time inference  
- ☁️ Cloud-based processing and alert system  
- 📱 Instant Telegram / SMS notifications for violent activity  
- 🧭 YOLO-based optimized face detection  
- 📊 Web dashboard for behavior analytics and visual insights  

---
