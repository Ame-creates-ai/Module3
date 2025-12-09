import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import os

# --- CONFIGURATION ---
# The app looks for this specific file in your folder
WATERMARK_FILENAME = "watermark.jpg" 

# --- HELPER: WATERMARK PREPARATION (Follows E-Signature Notebook Logic) ---
def prepare_watermark_opencv(watermark_bgr):
    """
    Prepares the watermark using the logic from Application_E_Signature.ipynb:
    1. Convert to Gray
    2. Threshold to create a mask (Remove White Background)
    3. Split Channels
    4. Merge with Alpha
    """
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(watermark_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Threshold & Invert (Standard E-Signature Logic)
    # We assume the signature/watermark is dark ink on light background.
    # Threshold: Low values (ink) become 0, High values (paper) become 255.
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Invert: Ink becomes 255 (Opaque), Paper becomes 0 (Transparent)
    alpha_mask = cv2.bitwise_not(thresh)
    
    # 3. Split Channels
    b, g, r = cv2.split(watermark_bgr)
    
    # 4. Merge Alpha (The "Code you gave me")
    watermark_bgra = cv2.merge((b, g, r, alpha_mask))
    
    return watermark_bgra

def apply_bottom_right_watermark(base_img_bgr):
    """
    Applies the prepared watermark to the bottom right of the base image.
    """
    # Check if watermark file exists
    if not os.path.exists(WATERMARK_FILENAME):
        # Fallback if file is missing (Just return original to prevent crash)
        cv2.putText(base_img_bgr, "watermark.jpg missing", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return base_img_bgr

    # Load Watermark using OpenCV
    wm_src = cv2.imread(WATERMARK_FILENAME)
    if wm_src is None: return base_img_bgr
    
    # Prepare it (Add Alpha/Transparency)
    wm_bgra = prepare_watermark_opencv(wm_src)
    
    # Convert Base to BGRA (to handle transparency compositing)
    if base_img_bgr.shape[2] == 3:
        base_bgra = cv2.cvtColor(base_img_bgr, cv2.COLOR_BGR2BGRA)
    else:
        base_bgra = base_img_bgr.copy()

    # --- RESIZE WATERMARK (20% of Main Image Width) ---
    h_img, w_img = base_bgra.shape[:2]
    h_wm, w_wm = wm_bgra.shape[:2]
    
    scale_ratio = (w_img * 0.20) / w_wm
    new_w = int(w_wm * scale_ratio)
    new_h = int(h_wm * scale_ratio)
    
    if new_w > 0 and new_h > 0:
        wm_resized = cv2.resize(wm_bgra, (new_w, new_h))
        
        # --- OVERLAY LOGIC (Region of Interest) ---
        # Position: Bottom Right with 10px padding
        y_offset = h_img - new_h - 10
        x_offset = w_img - new_w - 10
        
        # Ensure ROI is within bounds
        if y_offset >= 0 and x_offset >= 0:
            roi = base_bgra[y_offset:y_offset+new_h, x_offset:x_offset+new_w]
            
            # Normalize alpha to 0-1 range for math
            alpha_wm = wm_resized[:, :, 3] / 255.0
            alpha_bg = 1.0 - alpha_wm
            
            # Blend channels
            for c in range(0, 3):
                roi[:, :, c] = (alpha_wm * wm_resized[:, :, c] + alpha_bg * roi[:, :, c])
            
            # Put back into main image
            base_bgra[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = roi

    return base_bgra

def convert_to_bytes(img_bgra):
    # Convert BGRA (OpenCV) to RGB (PIL) for saving
    img_rgb = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGBA)
    pil_img = Image.fromarray(img_rgb)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

# --- TAB 1: THRESHOLDING (Notebook 2-2) ---
def tab_thresholding():
    st.header("Thresholding (Notebook 2-2)")
    uploaded_file = st.file_uploader("Upload Image", key="thresh")
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # User Controls
        mode = st.radio("Method", ["Binary", "Adaptive Mean", "Otsu"])
        
        if mode == "Binary":
            val = st.slider("Threshold", 0, 255, 127)
            _, processed = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY)
        elif mode == "Adaptive Mean":
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        else:
            _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        # Convert grayscale result back to BGR for watermarking consistency
        processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        
        # Apply Watermark
        final = apply_bottom_right_watermark(processed_bgr)
        
        # Display
        st.image(cv2.cvtColor(final, cv2.COLOR_BGRA2RGBA), caption="Result + Watermark", use_column_width=True)
        st.download_button("Download", convert_to_bytes(final), "threshold_watermarked.png", "image/png")

# --- TAB 2: LOGICAL OPERATIONS (Notebook 2-3) ---
def tab_logical():
    st.header("Logical Operations (Notebook 2-3)")
    uploaded_file = st.file_uploader("Upload Image", key="logic")
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        # Create Mask (Follows notebook logic of mask creation)
        mask = np.zeros(img.shape[:2], dtype="uint8")
        h, w = mask.shape
        shape = st.radio("Mask Shape", ["Circle", "Rectangle"])
        
        if shape == "Circle":
            cv2.circle(mask, (w//2, h//2), min(h,w)//3, 255, -1)
        else:
            cv2.rectangle(mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)
            
        # Apply Bitwise AND (The core notebook concept)
        processed = cv2.bitwise_and(img, img, mask=mask)
        
        # Apply Watermark
        final = apply_bottom_right_watermark(processed)
        
        c1, c2 = st.columns(2)
        c1.image(mask, caption="Mask", use_column_width=True)
        c2.image(cv2.cvtColor(final, cv2.COLOR_BGRA2RGBA), caption="Result + Watermark", use_column_width=True)
        st.download_button("Download", convert_to_bytes(final), "logical_watermarked.png", "image/png")

# --- TAB 3: ALPHA CHANNEL (Notebook 2-4) ---
def tab_alpha():
    st.header("Alpha Channel (Notebook 2-4)")
    uploaded_file = st.file_uploader("Upload Image", key="alpha")
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        # Notebook Logic: Split -> Create Alpha -> Merge
        b, g, r = cv2.split(img)
        
        # Interactive slider for the alpha value
        alpha_val = st.slider("Transparency Level", 0, 255, 128)
        
        # Create alpha channel array
        alpha_channel = np.ones(b.shape, dtype=b.dtype) * alpha_val
        
        # Merge (The code you gave me logic)
        img_bgra = cv2.merge((b, g, r, alpha_channel))
        
        # Apply Watermark (Note: img_bgra already has alpha, function handles it)
        final = apply_bottom_right_watermark(img_bgra)
        
        st.image(cv2.cvtColor(final, cv2.COLOR_BGRA2RGBA), caption="Result + Watermark", use_column_width=True)
        st.download_button("Download", convert_to_bytes(final), "alpha_watermarked.png", "image/png")

# --- MAIN APP LAYOUT ---
def main():
    st.set_page_config(page_title="OpenCV Tools", layout="wide")
    
    # Check for watermark file
    if not os.path.exists(WATERMARK_FILENAME):
        st.error(f"⚠️ Warning: '{WATERMARK_FILENAME}' not found. Please upload it to your folder for the watermark to work.")

    # Tabs for the 3 Notebook Requirements
    tab1, tab2, tab3 = st.tabs(["Thresholding", "Logical Operations", "Alpha Channel"])
    
    with tab1:
        tab_thresholding()
    with tab2:
        tab_logical()
    with tab3:
        tab_alpha()

if __name__ == "__main__":
    main()
