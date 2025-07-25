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


# ✅ Page config
st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ✅ Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .stButton > button {
        color: black;
        background-color: #000000;
        border-radius: 6px;
        font-size: 1rem;
    }
    .stSidebar, .css-1d391kg {
        color: black !important;
        font-size: 1rem;
    }
    .prediction {
        font-size: 1.3rem;
        color: #000000;
        font-weight: 600;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #444;
    }
    .footer {
        font-size: 0.9rem;
        color: #888;
        margin-top: 3em;
    }
    </style>
""", unsafe_allow_html=True)

# ✅ Cached model loader
@st.cache_resource
def load_trained_model():
    print("🔁 load_trained_model() CALLED")
    try:
        if not os.path.exists(MODEL_PATH):
            st.warning(f"⚠️ Model not found at: {MODEL_PATH}")
            return None

        print(f"📂 Model exists at: {MODEL_PATH}")
        model = load_model(MODEL_PATH, NUM_CLASSES)

        model.eval()
        print("✅ Model loaded successfully")
        st.success("✅ Model loaded.")
        return model
    except Exception as e:
        print("❌ Error in load_trained_model():", str(e))
        st.error(f"Error loading model: {e}")
        return None

# ✅ Main Streamlit app
def main():
    print("🧠 main() running...")

    model = load_trained_model()
    if model is None:
        st.error("Cannot proceed without a model. Please check the issues above.")
        st.stop()

    st.markdown(
        """
        <h2 style='color:#f5f7fa; text-align:center; background-color:#1e1e1e; padding:10px; border-radius:8px;'>
        🔬 Skin Lesion Classifier with GradCAM Explainability
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="subtitle">Upload a skin lesion image. Get an instant AI prediction and see what areas influenced the model\'s decision.</div>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("Instructions")
    st.sidebar.write("""
    1. Upload a dermatoscopic skin image.
    2. Wait for the model to process and classify.
    3. See the top predicted diagnosis and heatmap explanation.
    """)
    st.sidebar.markdown("---")
    st.sidebar.write("**Note:** This demo is for educational purposes only. Not for diagnostic use.")
    st.write("Model path exists?", os.path.exists(MODEL_PATH))

    # File upload
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

                st.markdown(
                    f'<div class="prediction">Prediction: {CLASS_NAMES[pred_class].upper()}'
                    f' <span style="color:#222;font-weight:400;">({confidence:.2%} confidence)</span></div>',
                    unsafe_allow_html=True
                )

                st.write("**Class probabilities:**")
                prob_dict = {CLASS_NAMES[i]: pred_prob[i].item() for i in range(len(CLASS_NAMES))}
                st.bar_chart(prob_dict)

                if st.checkbox("Show GradCAM Explanation"):
                    with st.spinner("Generating explanation..."):
                        gradcam_img = get_gradcam(model, input_tensor, pred_class)
                        st.image(gradcam_img, caption="GradCAM Heatmap", use_column_width=True)
    else:
        st.info("Upload a JPG or PNG image of a skin lesion to begin.")

    st.markdown('<div class="footer">Model for demonstrative purposes. Develop responsibly in all medical AI projects.</div>', unsafe_allow_html=True)

# ✅ Run app
if __name__ == "__main__":
    print("🧠 Running main() now!")
    main()
