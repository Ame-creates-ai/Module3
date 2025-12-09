import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import os
import requests

# --- CONFIGURATION ---
# We point to the RAW version of the file on GitHub so we can download it
WATERMARK_URL = "https://raw.githubusercontent.com/Ame-creates-ai/Module3/main/Module3/MySignature.jpg"

# --- HELPER: DOWNLOAD WATERMARK ---
@st.cache_resource
def download_watermark(url):
    """
    Downloads the watermark image from GitHub.
    Returns the image in OpenCV format (BGR).
    """
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Convert bytes to numpy array
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            # Decode to OpenCV image
            img_bgr = cv2.imdecode(image_array, 1)
            return img_bgr
        else:
            return None
    except Exception as e:
        return None

# --- HELPER: WATERMARK PREPARATION (Notebook Logic) ---
def prepare_watermark_opencv(watermark_bgr):
    """
    Prepares the watermark using Application_E_Signature.ipynb logic:
    1. Convert to Gray
    2. Threshold (Remove White Background)
    3. Split Channels
    4. Merge with Alpha
    """
    if watermark_bgr is None: return None
    
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(watermark_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Threshold & Invert
    # Threshold: Low values (ink) -> 0, High values (paper) -> 255
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # Invert: Ink -> 255 (Opaque), Paper -> 0 (Transparent)
    alpha_mask = cv2.bitwise_not(thresh)
    
    # 3. Split Channels
    b, g, r = cv2.split(watermark_bgr)
    
    # 4. Merge Alpha
    watermark_bgra = cv2.merge((b, g, r, alpha_mask))
    
    return watermark_bgra

def apply_bottom_right_watermark(base_img_bgr):
    """
    Applies the downloaded GitHub watermark to the bottom right.
    """
    # Download/Load Watermark
    wm_src = download_watermark(WATERMARK_URL)
    
    if wm_src is None:
        # Fallback text if download fails
        cv2.putText(base_img_bgr, "Watermark Fetch Failed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return base_img_bgr
    
    # Prepare it (Transparency)
    wm_bgra = prepare_watermark_opencv(wm_src)
    
    # Convert Base to BGRA
    if base_img_bgr.shape[2] == 3:
        base_bgra = cv2.cvtColor(base_img_bgr, cv2.COLOR_BGR2BGRA)
    else:
        base_bgra = base_img_bgr.copy()

    # --- RESIZE LOGIC (20% of Main Image Width) ---
    h_img, w_img = base_bgra.shape[:2]
    h_wm, w_wm = wm_bgra.shape[:2]
    
    scale_ratio = (w_img * 0.20) / w_wm
    new_w = int(w_wm * scale_ratio)
    new_h = int(h_wm * scale_ratio)
    
    if new_w > 0 and new_h > 0:
        wm_resized = cv2.resize(wm_bgra, (new_w, new_h))
        
        # --- POSITION LOGIC ---
        # Bottom Right with padding
        y_offset = h_img - new_h - 10
        x_offset = w_img - new_w - 10
        
        # Overlay
        if y_offset >= 0 and x_offset >= 0:
            roi = base_bgra[y_offset:y_offset+new_h, x_offset:x_offset+new_w]
            
            # Normalize alpha (0-1)
            alpha_wm = wm_resized[:, :, 3] / 255.0
            alpha_bg = 1.0 - alpha_wm
            
            # Blend
            for c in range(0, 3):
                roi[:, :, c] = (alpha_wm * wm_resized[:, :, c] + alpha_bg * roi[:, :, c])
            
            # Update main image
            base_bgra[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = roi

    return base_bgra

def convert_to_bytes(img_bgra):
    img_rgb = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGBA)
    pil_img = Image.fromarray(img_rgb)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

# --- TAB 1: THRESHOLDING ---
def tab_thresholding():
    st.header("Thresholding")
    uploaded_file = st.file_uploader("Upload Image", key="thresh")
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        mode = st.radio("Method", ["Binary", "Adaptive Mean", "Otsu"])
        
        if mode == "Binary":
            val = st.slider("Threshold", 0, 255, 127)
            _, processed = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY)
        elif mode == "Adaptive Mean":
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        else:
            _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        final = apply_bottom_right_watermark(processed_bgr)
        st.image(cv2.cvtColor(final, cv2.COLOR_BGRA2RGBA), use_column_width=True)
        st.download_button("Download", convert_to_bytes(final), "threshold.png", "image/png")

# --- TAB 2: LOGICAL OPERATIONS ---
def tab_logical():
    st.header("Logical Operations")
    uploaded_file = st.file_uploader("Upload Image", key="logic")
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        mask = np.zeros(img.shape[:2], dtype="uint8")
        h, w = mask.shape
        shape = st.radio("Mask Shape", ["Circle", "Rectangle"])
        
        if shape == "Circle":
            cv2.circle(mask, (w//2, h//2), min(h,w)//3, 255, -1)
        else:
            cv2.rectangle(mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)
            
        processed = cv2.bitwise_and(img, img, mask=mask)
        final = apply_bottom_right_watermark(processed)
        
        c1, c2 = st.columns(2)
        c1.image(mask, caption="Mask", use_column_width=True)
        c2.image(cv2.cvtColor(final, cv2.COLOR_BGRA2RGBA), caption="Result", use_column_width=True)
        st.download_button("Download", convert_to_bytes(final), "logical.png", "image/png")

# --- TAB 3: ALPHA CHANNEL ---
def tab_alpha():
    st.header("Alpha Channel")
    uploaded_file = st.file_uploader("Upload Image", key="alpha")
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        b, g, r = cv2.split(img)
        
        alpha_val = st.slider("Transparency Level", 0, 255, 128)
        alpha_channel = np.ones(b.shape, dtype=b.dtype) * alpha_val
        img_bgra = cv2.merge((b, g, r, alpha_channel))
        
        final = apply_bottom_right_watermark(img_bgra)
        st.image(cv2.cvtColor(final, cv2.COLOR_BGRA2RGBA), use_column_width=True)
        st.download_button("Download", convert_to_bytes(final), "alpha.png", "image/png")

# --- MAIN ---
def main():
    st.set_page_config(page_title="OpenCV Interactive", layout="wide")
    
    # Test connection to watermark
    wm_test = download_watermark(WATERMARK_URL)
    if wm_test is None:
        st.warning(f"⚠️ Could not verify watermark at: {WATERMARK_URL}. Check URL or Internet.")

    tab1, tab2, tab3 = st.tabs(["Thresholding", "Logical Operations", "Alpha Channel"])
    
    with tab1: tab_thresholding()
    with tab2: tab_logical()
    with tab3: tab_alpha()

if __name__ == "__main__":
    main()
