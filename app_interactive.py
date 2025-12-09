import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import requests

# --- CONFIGURATION ---
# CORRECT URL (Points to raw image data)
WATERMARK_URL = "https://raw.githubusercontent.com/Ame-creates-ai/Module3/main/Module3/MySignature.jpg"

# --- HELPER: DOWNLOAD WATERMARK ---
@st.cache_resource
def download_watermark(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            return cv2.imdecode(image_array, 1)
        return None
    except:
        return None

# --- HELPER: PREPARE WATERMARK (Notebook Logic) ---
def prepare_watermark_opencv(watermark_bgr):
    """
    Logic from Application_E_Signature.ipynb:
    1. Grayscale -> 2. Threshold (Invert) -> 3. Split -> 4. Merge Alpha
    """
    if watermark_bgr is None: return None
    
    # 1. Convert to Gray
    gray = cv2.cvtColor(watermark_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Threshold & Invert (Ink=255/Opaque, Paper=0/Transparent)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    alpha_mask = cv2.bitwise_not(thresh)
    
    # 3. Split Channels
    b, g, r = cv2.split(watermark_bgr)
    
    # 4. Merge with Alpha
    return cv2.merge((b, g, r, alpha_mask))

def apply_bottom_right_watermark(base_img_bgr):
    # Download & Prepare
    wm_src = download_watermark(WATERMARK_URL)
    if wm_src is None:
        # Fallback text if URL fails
        cv2.putText(base_img_bgr, "Sig Not Found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return base_img_bgr
    
    wm_bgra = prepare_watermark_opencv(wm_src)
    
    # Convert Base to BGRA
    if base_img_bgr.shape[2] == 3:
        base_bgra = cv2.cvtColor(base_img_bgr, cv2.COLOR_BGR2BGRA)
    else:
        base_bgra = base_img_bgr.copy()

    # Resize Watermark (25% of width)
    h_img, w_img = base_bgra.shape[:2]
    h_wm, w_wm = wm_bgra.shape[:2]
    
    scale = (w_img * 0.25) / w_wm
    new_w, new_h = int(w_wm * scale), int(h_wm * scale)
    
    if new_w > 0 and new_h > 0:
        wm_resized = cv2.resize(wm_bgra, (new_w, new_h))
        
        # Position: Bottom Right
        y = h_img - new_h - 10
        x = w_img - new_w - 10
        
        # Alpha Blend
        if y >= 0 and x >= 0:
            roi = base_bgra[y:y+new_h, x:x+new_w]
            alpha_wm = wm_resized[:, :, 3] / 255.0
            alpha_bg = 1.0 - alpha_wm
            
            for c in range(3):
                roi[:, :, c] = (alpha_wm * wm_resized[:, :, c] + alpha_bg * roi[:, :, c])
            
            # Update Alpha channel of ROI to be opaque where signature is
            roi[:, :, 3] = 255
            
            base_bgra[y:y+new_h, x:x+new_w] = roi

    return base_bgra

def convert_to_bytes(img_bgra):
    img_rgb = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGBA)
    buf = BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG")
    return buf.getvalue()

# --- APP TABS ---

def tab_thresholding():
    st.header("Thresholding")
    up = st.file_uploader("Upload Image", key="th")
    if up:
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        mode = st.radio("Mode", ["Binary", "Adaptive", "Otsu"])
        if mode == "Binary":
            v = st.slider("Val", 0, 255, 127)
            _, res = cv2.threshold(gray, v, 255, cv2.THRESH_BINARY)
        elif mode == "Adaptive":
            res = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        else:
            _, res = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            
        final = apply_bottom_right_watermark(cv2.cvtColor(res, cv2.COLOR_GRAY2BGR))
        st.image(cv2.cvtColor(final, cv2.COLOR_BGRA2RGBA), use_column_width=True)
        st.download_button("Download", convert_to_bytes(final), "thresh.png", "image/png")

def tab_logical():
    st.header("Logical Operations")
    up = st.file_uploader("Upload Image", key="lg")
    if up:
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), 1)
        mask = np.zeros(img.shape[:2], dtype="uint8")
        h, w = mask.shape
        
        shape = st.radio("Mask", ["Circle", "Rectangle"])
        if shape == "Circle": cv2.circle(mask, (w//2, h//2), min(h,w)//3, 255, -1)
        else: cv2.rectangle(mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)
            
        res = cv2.bitwise_and(img, img, mask=mask)
        final = apply_bottom_right_watermark(res)
        
        c1, c2 = st.columns(2)
        c1.image(mask, caption="Mask", use_column_width=True)
        c2.image(cv2.cvtColor(final, cv2.COLOR_BGRA2RGBA), caption="Result", use_column_width=True)
        st.download_button("Download", convert_to_bytes(final), "logical.png", "image/png")

def tab_alpha():
    st.header("Alpha Channel")
    up = st.file_uploader("Upload Image", key="al")
    if up:
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), 1)
        b,g,r = cv2.split(img)
        alpha = st.slider("Alpha", 0, 255, 128)
        
        # Create Alpha Channel & Merge
        a_chan = np.ones(b.shape, dtype=b.dtype) * alpha
        res = cv2.merge((b,g,r,a_chan))
        
        final = apply_bottom_right_watermark(res)
        st.image(cv2.cvtColor(final, cv2.COLOR_BGRA2RGBA), use_column_width=True)
        st.download_button("Download", convert_to_bytes(final), "alpha.png", "image/png")

# --- MAIN ---
def main():
    st.set_page_config(layout="wide")
    t1, t2, t3 = st.tabs(["Thresholding", "Logical Operations", "Alpha Channel"])
    with t1: tab_thresholding()
    with t2: tab_logical()
    with t3: tab_alpha()

if __name__ == "__main__":
    main()
