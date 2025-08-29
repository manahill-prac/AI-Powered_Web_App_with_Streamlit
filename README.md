# 🐾 Cats vs Dogs Classifier – AI-Powered Web App

![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit)
![TensorFlow Lite](https://img.shields.io/badge/Model-TensorFlow%20Lite-orange?logo=tensorflow)
![Status](https://img.shields.io/badge/Status-Deployed-brightgreen)

##  Overview

This project is an **AI-powered image classifier** that predicts whether an uploaded picture is of a **Cat** or a **Dog**.
The app is built using **Streamlit** and powered by a **MobileNetV2 model fine-tuned on the Cats vs Dogs dataset**, converted to **TensorFlow Lite (TFLite)** for lightweight deployment.

🔗 Try the Live App: *\[Streamlit Cloud Link]*

---
##  Installation & Local Run

1. Clone this repository:

   ```bash
   git clone https://github.com/<your-username>/cats-dogs-classifier.git
   cd cats-dogs-classifier
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

---
## Model Details

* Base Model: **MobileNetV2** (transfer learning)
* Fine-Tuned on: **Cats vs Dogs dataset (\~37k images)**
* Format: Converted from `.keras` → `.tflite` for deployment
* Accuracy: \~97-98%

---

