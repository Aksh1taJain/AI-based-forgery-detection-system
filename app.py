import streamlit as st
import requests
from PIL import Image

# ----------------------------
# CONFIG
# ----------------------------
API_URL = "http://127.0.0.1:5000/predict"   # Change if backend is hosted remotely

st.title("🕵️‍♂️ Image Forgery Detection System")
st.write("""
Upload an image and the system will:
- Detect if the image is **AI-generated or real**  
- Check if a **real image has forgery signs (Copy-Move/Splicing)**  
""")

uploaded_file = st.file_uploader("📎 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=350)

    if st.button("🔍 Analyze Image"):
        with st.spinner("⏳ Processing...please wait..."):
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                result = response.json()

                # -------------------------
                # Extract Results
                # -------------------------
                forgery_pred = result['forgery_analysis']['prediction']
                forgery_conf = result['forgery_analysis']['confidence']

                ai_pred = result['ai_image_analysis']['prediction']
                ai_conf = result['ai_image_analysis']['confidence']

                # -------------------------
                # Display Styled Output
                # -------------------------
                st.subheader("📌 AI vs Real Image Detection")
                st.write(f"**Result:** `{ai_pred}`")
                st.write(f"🔹 Confidence: **{ai_conf}%**")

                st.subheader("🛠 Forgery Detection Analysis")
                st.write(f"**Result:** `{forgery_pred}`")
                st.write(f"🔹 Confidence: **{forgery_conf * 100:.2f}%**" if isinstance(forgery_conf, float) else forgery_conf)

            else:
                st.error("❌ Prediction failed. Please ensure Flask server is running.")
