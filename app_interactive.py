import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
from io import BytesIO
import os

# --- CONFIGURATION ---
# Replace this with your actual watermark filename in your GitHub/Folder
WATERMARK_FILENAME = "watermark.jpg" 

# --- HELPER FUNCTIONS ---

def convert_image_to_bytes(pil_img, format="PNG"):
    buf = BytesIO()
    pil_img.save(buf, format=format)
    return buf.getvalue()

def load_github_watermark():
    """
    Tries to load the watermark image from the local file system.
    Returns None if file is not found.
    """
    if os.path.exists(WATERMARK_FILENAME):
        return Image.open(WATERMARK_FILENAME).convert("RGBA")
    return None

def apply_watermark(base_image):
    """
    Applies the 'GitHub Watermark' (watermark.jpg) if found.
    Otherwise, applies a default text watermark.
    """
    wm_img = load_github_watermark()
    
    if wm_img:
        # Resize watermark to 20% of base image width
        target_width = int(base_image.width * 0.20)
        aspect_ratio = wm_img.height / wm_img.width
        target_height = int(target_width * aspect_ratio)
        
        # Resize
        wm_resized = wm_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        # Position: Bottom Right
        margin = 20
        x = base_image.width - wm_resized.width - margin
        y = base_image.height - wm_resized.height - margin
        
        # Composite
        watermarked = base_image.copy()
        watermarked.paste(wm_resized, (x, y), wm_resized)
        return watermarked
    else:
        # Fallback Text Watermark if 'watermark.jpg' is missing
        draw = ImageDraw.Draw(base_image)
        try:
            font_size = int(base_image.height * 0.05)
            font = ImageFont.truetype("arial.ttf", max(font_size, 20))
        except:
            font = ImageFont.load_default()
        
        text = "WATERMARK.JPG NOT FOUND"
        
        # Position text bottom right
        # (Simplified positioning for fallback)
        w, h = base_image.size
        x, y = w - 300, h - 50
        
        # Draw Shadow/Outline
        draw.text((x+2, y+2), text, font=font, fill="black")
        draw.text((x, y), text, font=font, fill="white")
        
        return base_image

# --- PAGE: THRESHOLDING (Notebook 2-2) ---
def page_thresholding():
    st.title("Interactive Thresholding")
    st.write("Explore how computers separate objects from backgrounds.")
    
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'], key="thresh")
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        c1, c2 = st.columns(2)
        with c1: st.image(img_bgr, caption="Original", channels="BGR", use_column_width=True)
        
        st.sidebar.header("Threshold Settings")
        method = st.sidebar.radio("Method", ["Global Binary", "Adaptive Mean", "Otsu's Binarization"])
        
        if method == "Global Binary":
            val = st.sidebar.slider("Threshold Value", 0, 255, 127)
            _, res = cv2.threshold(img_gray, val, 255, cv2.THRESH_BINARY)
            
        elif method == "Adaptive Mean":
            block = st.sidebar.slider("Block Size", 3, 51, 11, step=2)
            C = st.sidebar.slider("Constant", 0, 20, 2)
            res = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, C)
            
        elif method == "Otsu's Binarization":
            val, res = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            st.sidebar.info(f"Otsu Value: {val}")

        # Convert result to PIL for water
