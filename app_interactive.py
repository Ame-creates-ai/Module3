import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import requests

# --- 1. CONFIGURATION ---
# Direct link to your signature on GitHub
WATERMARK_URL = "https://raw.githubusercontent.com/Ame-creates-ai/Module3/main/Module3/MySignature.jpg"

# --- 2. INTEGRATION: YOUR WATERMARK LOGIC ---
def process_watermark(sig_bgr):
    """
    INTEGRATED LOGIC from your 'application_e_signature.py':
    1. Convert to Gray
    2. Create Mask (Threshold & Invert)
    3. Split Channels
    4. Merge with Alpha
    """
    if sig_bgr is None: return None
    
    # [From your code]: Convert to grayscale
    gray = cv2.cvtColor(sig_bgr, cv2.COLOR_BGR2GRAY)
    
    # [From your code]: Threshold to create binary mask
    # You used standard binary thresholding logic in your notebook
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # [From your code]: Create inverse mask (Ink becomes opaque/255)
    alpha_mask = cv2.bitwise_not(thresh)
    
    # [From your code]: Split the color channels
    b, g, r = cv2.split(sig_bgr)
    
    # [From your code]: Merge channels with the new alpha mask
    sig_bgra = cv2.merge((b, g, r, alpha_mask))
    
    return sig_bgra

# --- 3. HELPER FUNCTIONS ---
@st.cache_resource
def get_watermark_from_github():
    try:
        response = requests.get(WATERMARK_URL)
        if response.status_code == 200:
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            return cv2.imdecode(image_array, 1)
    except:
        return None
    return None

def apply_watermark_to_image(main_img):
    # Get watermark
    raw_sig = get_watermark_from_github()
    if raw_sig is None:
        cv2.putText(main_img, "Sig Not Found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return main_img

    # Process using your integrated logic
    wm_bgra = process_watermark(raw_sig)
    
    # Prepare Main Image
    if main_img.shape[2] == 3:
        main_bgra = cv2.cvtColor(main_img, cv2.COLOR_BGR2BGRA)
    else:
        main_bgra = main_img.copy()
        
    # Resize Watermark (25% width)
    h_img, w_img = main_bgra.shape[:2]
    h_wm, w_wm = wm_bgra.shape[:2]
    scale = (w_img * 0.25) / w_wm
    new_w, new_h = int(w_wm * scale), int(h_wm * scale)
    
    if new_w > 0 and new_h > 0:
        wm_resized = cv2.resize(wm_bgra, (new_w, new_h))
        
        # Position: Bottom Right
        y_pos = h_img - new_h - 20
        x_pos = w_img - new_w - 20
        
        # Overlay
        if y_pos >= 0 and x_pos >= 0:
            roi = main_bgra[y_pos:y_pos+new_h, x_pos:x_pos+new_w]
            alpha_wm = wm_resized[:, :, 3] / 255.0
            alpha_bg = 1.0 - alpha_wm
            for c in range(3):
                roi[:, :, c] = (alpha_wm * wm_resized[:, :, c] + alpha_bg * roi[:, :, c])
            roi[:, :, 3] = 255
            main_bgra[y_pos:y_pos+new_h, x_pos:x_pos+new_w] = roi
            
    return main_bgra

def convert_to_bytes(img_bgra):
    img_rgb = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGBA)
    buf = BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG")
    return buf.getvalue()

# --- 4. APP INTERFACE ---
def main():
    st.set_page_config(layout="wide", page_title="Module 3 App")
    
    t1, t2, t3 = st.tabs(["Thresholding", "Logical Operations", "Alpha Channel"])

    # --- TAB 1: Thresholding ---
    with t1:
        st.header("Thresholding")
        up = st.file_uploader("Upload Image", key="t")
        if up:
            img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mode = st.radio("Method", ["Binary", "Adaptive Mean", "Otsu"])
            
            if mode == "Binary":
                v = st.slider("Val", 0, 255, 127)
                _, res = cv2.threshold(gray, v, 255, cv2.THRESH_BINARY)
            elif mode == "Adaptive Mean":
                res = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            else:
                _, res = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            
            # Convert to color so we can see the colored signature
            res_color = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
            final = apply_watermark_to_image(res_color)
            st.image(cv2.cvtColor(final, cv2.COLOR_BGRA2RGBA), use_column_width=True)
            st.download_button("Download", convert_to_bytes(final), "thresh.png", "image/png")

    # --- TAB 2: Logical Operations ---
    with t2:
        st.header("Logical Operations")
        up = st.file_uploader("Upload Image", key="l")
        if up:
            img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), 1)
            mask = np.zeros(img.shape[:2], dtype="uint8")
            h, w = mask.shape
            shape = st.radio("Mask Shape", ["Circle", "Rectangle"])
            
            if shape == "Circle": cv2.circle(mask, (w//2, h//2), min(h,w)//3, 255, -1)
            else: cv2.rectangle(mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)
            
            res = cv2.bitwise_and(img, img, mask=mask)
            final = apply_watermark_to_image(res)
            
            c1, c2 = st.columns(2)
            c1.image(mask, caption="Mask", use_column_width=True)
            c2.image(cv2.cvtColor(final, cv2.COLOR_BGRA2RGBA), caption="Result", use_column_width=True)
            st.download_button("Download", convert_to_bytes(final), "logical.png", "image/png")

    # --- TAB 3: Alpha Channel ---
    with t3:
        st.header("Alpha Channel")
        up = st.file_uploader("Upload Image", key="a")
        if up:
            img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), 1)
            b,g,r = cv2.split(img)
            alpha = st.slider("Alpha", 0, 255, 128)
            a_chan = np.ones(b.shape, dtype=b.dtype) * alpha
            res = cv2.merge((b,g,r,a_chan))
            
            final = apply_watermark_to_image(res)
            st.image(cv2.cvtColor(final, cv2.COLOR_BGRA2RGBA), use_column_width=True)
            st.download_button("Download", convert_to_bytes(final), "alpha.png", "image/png")

if __name__ == "__main__":
    main()
