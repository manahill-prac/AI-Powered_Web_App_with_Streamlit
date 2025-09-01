import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite


# ----------------- CONFIG -----------------
st.set_page_config(page_title="Cats vs Dogs Classifier",
                   page_icon="🐾",
                   layout="wide")


PRIMARY_COLOR = "#20B2AA"

# ----------------- NAVIGATION -----------------
PAGES = ["Home", "Demo", "About", "Credits"]
choice = st.sidebar.radio("Navigation", PAGES)


# ----------------- LOAD MODEL -----------------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="trained_model/cats_dogs_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# Get input & output details (needed for predictions)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ["Cat", "Dog"]


# ----------------- PREDICTION FUNCTION -----------------
def predict(image: Image.Image):
    img_resized = image.resize((160, 160))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0][0]

    label = class_names[int(preds > 0.5)]
    confidence = preds if label == "Dog" else 1 - preds
    return label, float(confidence)


# ----------------- HOME -----------------
if choice == "Home":
    st.markdown(f"<h1 style='color:{PRIMARY_COLOR};'>🐾 Cats vs Dogs Classifier</h1>", unsafe_allow_html=True)
    st.write("Upload an image of a **Cat or Dog** and our AI model will predict the result with confidence.")
    st.image("https://placekitten.com/800/300", use_column_width=True)


# ----------------- DEMO -----------------
elif choice == "Demo":
    st.header("📷 Try the Demo")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        
        label, confidence = predict(image)

        st.subheader(f"Prediction: {label}")
        st.progress(confidence)

        col1, col2 = st.columns(2)
        col1.metric("Confidence", f"{confidence:.2%}")
        col2.metric("Class", label)


# ----------------- ABOUT -----------------
elif choice == "About":
    st.header("ℹ️ About this App")
    st.write("This AI-powered web app uses **Transfer Learning with MobileNetV2** trained on the **Cats vs Dogs dataset**.")
    st.write("- Model Accuracy: ~98%")
    st.write("- Frameworks: TensorFlow Lite (via TensorFlow) + Streamlit")


# ----------------- CREDITS -----------------
elif choice == "Credits":
    st.header("👩‍💻 Credits")
    st.write("Developed by **Manahil** as part of AI/ML learning projects.")
    st.write("Powered by TensorFlow Lite & Streamlit.")
