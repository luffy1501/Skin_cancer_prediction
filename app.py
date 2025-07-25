import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import streamlit as st
import torch
from PIL import Image
from utils import load_model, preprocess_image, get_gradcam
from config import MODEL_PATH, CLASS_NAMES
from models.cnn_model import create_resnet18_model

MODEL_PATH = "models/saved_model.pth"
NUM_CLASSES = 7

# Lesion class descriptions & severity
CLASS_INFO = {
    "akiec": {
        "name": "Actinic Keratoses and Intraepithelial Carcinoma",
        "desc": "Potentially precancerous lesion often caused by sun damage. Should be evaluated by a dermatologist.",
        "serious": True,
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "desc": "A common skin cancer. Rarely spreads but needs treatment.",
        "serious": True,
    },
    "bkl": {
        "name": "Benign Keratosis-like Lesions",
        "desc": "Non-cancerous growths. Usually harmless.",
        "serious": False,
    },
    "df": {
        "name": "Dermatofibroma",
        "desc": "Benign skin growth, often due to insect bites or minor injuries.",
        "serious": False,
    },
    "mel": {
        "name": "Melanoma",
        "desc": "A serious form of skin cancer. Requires immediate medical attention.",
        "serious": True,
    },
    "nv": {
        "name": "Melanocytic Nevi",
        "desc": "Common moles. Usually benign but should be monitored.",
        "serious": False,
    },
    "vasc": {
        "name": "Vascular Lesions",
        "desc": "Blood vessel growths, often harmless, but some may need evaluation.",
        "serious": False,
    }
}

# Streamlit page config
st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #121212; color: #eeeeee; }
    .stButton > button {
        color: white;
        background-color: #333333;
        border-radius: 6px;
        font-size: 1rem;
    }
    .stSidebar, .css-1d391kg {
        color: white !important;
        font-size: 1rem;
    }
    .prediction {
        font-size: 1.3rem;
        color: #00ffcc;
        font-weight: 600;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #cccccc;
    }
    .footer {
        font-size: 0.9rem;
        color: #aaaaaa;
        margin-top: 3em;
    }
    </style>
""", unsafe_allow_html=True)


# Cached model loader
@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        st.warning(f"⚠️ Model not found at: {MODEL_PATH}")
        return None

    model = load_model(MODEL_PATH, NUM_CLASSES)
    model.eval()
    st.success("✅ Model loaded.")
    return model

# Main Streamlit app
def main():
    model = load_trained_model()
    if model is None:
        st.error("Cannot proceed without a model.")
        st.stop()

    st.markdown(
        """
        <h2 style='color:#f5f7fa; text-align:center; background-color:#1e1e1e; padding:10px; border-radius:8px;'>
        🔬 Skin Lesion Classifier with GradCAM Explainability
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="subtitle">Upload a skin lesion image. Get an AI prediction with a GradCAM explanation and medical insight.</div>', unsafe_allow_html=True)

    st.sidebar.header("Instructions")
    st.sidebar.write("""
    1. Upload a dermatoscopic skin image.
    2. View the predicted class and heatmap.
    3. Use the information as educational reference only.
    """)
    st.sidebar.markdown("---")
    st.sidebar.write("**Note:** This demo is for educational purposes only. Not for diagnostic use.")

    uploaded_file = st.file_uploader("📤 Upload a skin lesion image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        col1, col2 = st.columns([1, 2])

        with col1:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Input Image", use_column_width=True)

        with col2:
            input_tensor = preprocess_image(image)
            with torch.no_grad():
                output = model(input_tensor)
                pred_prob = torch.softmax(output, dim=1)[0]
                pred_class = pred_prob.argmax().item()
                confidence = pred_prob[pred_class].item()
                class_name = CLASS_NAMES[pred_class]

                st.markdown(
                    f'<div class="prediction">Prediction: {class_name.upper()} '
                    f'({confidence:.2%} confidence)</div>',
                    unsafe_allow_html=True
                )

                info = CLASS_INFO[class_name]
                st.markdown(f"**Meaning:** {info['name']}")
                st.markdown(f"**Details:** {info['desc']}")

                if info["serious"]:
                    st.error("⚠️ This type may require urgent medical evaluation.")

                st.write("**Class probabilities:**")
                prob_dict = {CLASS_NAMES[i]: pred_prob[i].item() for i in range(len(CLASS_NAMES))}
                st.bar_chart(prob_dict)

               
        st.info("Upload a JPG or PNG image of a skin lesion to begin.")

    st.markdown('<div class="footer">Model for research purposes only. Never substitute for clinical judgment.</div>', unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
