# ==========================================
# AI Banking Fraud Detection & Verification
# ==========================================

import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import pandas as pd
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# Try to import DeepFace safely
try:
    from deepface import DeepFace
    deepface_available = True
except Exception:
    deepface_available = False

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="AI Banking Verification Portal", layout="wide")

# ------------------- STYLING (Blue–Silver Theme) -------------------
st.markdown("""
<style>
/* ===== Base Page ===== */
body, .stApp {
    background: linear-gradient(180deg, #e9edf5 0%, #f7f8fb 50%, #dce3ed 100%);
    color: #0a142f; /* Deep navy text */
    font-family: 'Segoe UI', sans-serif;
}
header, footer {visibility: hidden;}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #002855 !important; /* Corporate navy blue */
    font-weight: 700;
}
p, li, span, div {
    color: #1a1a1a !important; /* readable blackish text */
}

/* ===== Sidebar Styling ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #001b44 0%, #003a70 100%) !important; /* Blue gradient */
}
[data-testid="stSidebar"] * {
    color: #e6eefc !important; /* Soft silver text */
}
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

/* Highlight selected radio option */
[data-testid="stSidebar"] [aria-checked="true"] {
    background: linear-gradient(90deg, #0b63d6, #00b4d8) !important;
    border-radius: 6px;
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* ===== Buttons ===== */
.stButton>button {
    background: linear-gradient(90deg, #004aad, #0078d7);
    color: #ffffff;
    font-weight: 600;
    border-radius: 10px;
    border: none;
    padding: 8px 18px;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #002855, #005fa3);
}

/* ===== Inputs, Tables, and Widgets ===== */
.stTextInput>div>div>input, textarea {
    color: #000000 !important;
    background-color: #ffffff !important;
    border-radius: 6px !important;
}

[data-testid="stDataFrame"] table {
    color: #0a142f !important;
}

/* ===== Section Container ===== */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ------------------- APP HEADER -------------------
st.title("🏦 AI Banking Fraud Detection & Verification System")
st.caption("A Secure AI-Powered System for Document, KYC, and Transaction Fraud Prevention")

st.sidebar.title("🔍 Fraud Detection Modules")
option = st.sidebar.radio("Select a Module", [
    "Dashboard Home",
    "Document Tampering",
    "Signature Verification",
    "Aadhaar Fraud Detection",
    "PAN Fraud Detection",
    "AI-Based KYC Verification",
    "Unusual Pattern Detection",
])

# ------------------- UTILITY FUNCTIONS -------------------
def compare_images(img1_bytes, img2_bytes):
    try:
        img1 = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        score, diff = ssim(img1, img2, full=True)
        diff = (diff * 255).astype("uint8")
        return round(score, 3), diff
    except Exception as e:
        st.error(f"Error comparing images: {e}")
        return None, None

def generate_pdf_report(title, result_text, score=None):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, 800, "AI Banking Fraud Detection Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, 770, f"Module: {title}")
    c.drawString(50, 750, f"Result: {result_text}")
    if score is not None:
        c.drawString(50, 730, f"Similarity Score: {score}")
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ------------------- DASHBOARD HOME -------------------
if option == "Dashboard Home":
    st.markdown("""
    <h3>🏛 Welcome to Secure Bank’s AI Verification Dashboard</h3>
    <p>Use this intelligent system to verify customer documents, detect potential frauds, and ensure compliance.</p>
    <ul>
    <li>📄 Document Forgery Detection</li>
    <li>✍ Signature Verification</li>
    <li>🪪 Aadhaar & PAN Validation</li>
    <li>🧬 AI-Based KYC Verification</li>
    <li>📊 Transaction Pattern Monitoring</li>
    </ul>
    """, unsafe_allow_html=True)

# ------------------- MODULE 1: DOCUMENT TAMPERING -------------------
elif option == "Document Tampering":
    st.header("📄 Document Forgery Detection")
    col1, col2 = st.columns(2)
    with col1:
        doc1 = st.file_uploader("Upload Original Document", type=["jpg", "png", "jpeg", "pdf"])
    with col2:
        doc2 = st.file_uploader("Upload Suspected Document", type=["jpg", "png", "jpeg", "pdf"])

    if doc1 and doc2:
        img1_bytes = doc1.read()
        img2_bytes = doc2.read()
        score, diff = compare_images(img1_bytes, img2_bytes)
        if score is not None:
            if score < 0.85:
                result_text = "⚠ Possible forgery detected."
                st.error(result_text)
            else:
                result_text = "✅ No significant alterations detected."
                st.success(result_text)
            pdf = generate_pdf_report("Document Tampering", result_text, score)
            st.download_button("📘 Download PDF Report", pdf, file_name="Document_Report.pdf")

# ------------------- MODULE 2: SIGNATURE VERIFICATION -------------------
elif option == "Signature Verification":
    st.header("✍ Signature Verification")
    col1, col2 = st.columns(2)
    with col1:
        sig1 = st.file_uploader("Upload Original Signature", type=["jpg", "png", "jpeg"])
    with col2:
        sig2 = st.file_uploader("Upload Submitted Signature", type=["jpg", "png", "jpeg"])

    if sig1 and sig2:
        sig1_img = cv2.imdecode(np.frombuffer(sig1.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        sig2_img = cv2.imdecode(np.frombuffer(sig2.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(sig1_img, None)
        kp2, des2 = orb.detectAndCompute(sig2_img, None)
        if des1 is not None and des2 is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            match_score = len(matches)
            if match_score > 50:
                result_text = "✅ Genuine Signature"
                st.success(result_text)
            else:
                result_text = "❌ Forged Signature Detected"
                st.error(result_text)
            pdf = generate_pdf_report("Signature Verification", result_text)
            st.download_button("📘 Download PDF Report", pdf, file_name="Signature_Report.pdf")

# ------------------- MODULE 3: AADHAAR FRAUD DETECTION -------------------
elif option == "Aadhaar Fraud Detection":
    st.header("🪪 Aadhaar Verification")
    aadhaar_num = st.text_input("Enter Aadhaar Number (XXXX-XXXX-XXXX):")
    if st.button("Verify Aadhaar"):
        if len(aadhaar_num) == 14:
            result_text = "✅ Aadhaar number format valid."
            st.success(result_text)
        else:
            result_text = "❌ Invalid Aadhaar format."
            st.error(result_text)
        pdf = generate_pdf_report("Aadhaar Fraud Detection", result_text)
        st.download_button("📘 Download PDF Report", pdf, file_name="Aadhaar_Report.pdf")

# ------------------- MODULE 4: PAN FRAUD DETECTION -------------------
elif option == "PAN Fraud Detection":
    st.header("💳 PAN Card Verification")
    pan = st.text_input("Enter PAN Number (ABCDE1234F):")
    if st.button("Validate PAN"):
        if len(pan) == 10 and pan[:5].isalpha() and pan[5:9].isdigit() and pan[-1].isalpha():
            result_text = "✅ Valid PAN Structure."
            st.success(result_text)
        else:
            result_text = "❌ Invalid PAN Format."
            st.error(result_text)
        pdf = generate_pdf_report("PAN Fraud Detection", result_text)
        st.download_button("📘 Download PDF Report", pdf, file_name="PAN_Report.pdf")

# ------------------- MODULE 5: AI-BASED KYC VERIFICATION -------------------
elif option == "AI-Based KYC Verification":
    st.header("🧬 AI-Powered KYC Face Verification")
    if not deepface_available:
        st.warning("⚠ DeepFace or TensorFlow not available. Install with:\n`pip install tensorflow>=2.16.0 keras>=2.16.0 deepface`")
    else:
        col1, col2 = st.columns(2)
        with col1:
            selfie = st.file_uploader("Upload Selfie Photo", type=["jpg", "png", "jpeg"])
        with col2:
            id_photo = st.file_uploader("Upload ID Photo", type=["jpg", "png", "jpeg"])
        if selfie and id_photo:
            st.info("Running facial match verification using AI...")
            try:
                result = DeepFace.verify(np.array(Image.open(selfie)), np.array(Image.open(id_photo)))
                if result["verified"]:
                    result_text = "✅ KYC Face Match Successful"
                    st.success(result_text)
                else:
                    result_text = "❌ KYC Face Mismatch Detected"
                    st.error(result_text)
                pdf = generate_pdf_report("AI-Based KYC Verification", result_text)
                st.download_button("📘 Download PDF Report", pdf, file_name="KYC_Report.pdf")
            except Exception as e:
                st.error(f"Error during KYC verification: {e}")

# ------------------- MODULE 6: UNUSUAL PATTERN DETECTION -------------------
elif option == "Unusual Pattern Detection":
    st.header("📊 Unusual Transaction Pattern Detection")
    uploaded_file = st.file_uploader("Upload Transaction Data (CSV, Excel)", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            st.dataframe(data.head())
            z_scores = (data - data.mean()) / data.std()
            anomalies = data[(abs(z_scores) > 3).any(axis=1)]
            st.subheader("🔎 Detected Unusual Patterns:")
            st.dataframe(anomalies)
            result_text = f"Detected {len(anomalies)} unusual patterns."
            pdf = generate_pdf_report("Unusual Pattern Detection", result_text)
            st.download_button("📘 Download PDF Report", pdf, file_name="Pattern_Report.pdf")
        except Exception as e:
            st.error(f"Error reading file: {e}")