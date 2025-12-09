import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
from io import BytesIO

# --- HELPER FUNCTIONS ---

def add_watermark(pil_image, text="© STUDENT 2025"):
    """
    Adds a text watermark to the bottom right of the PIL image.
    """
    draw = ImageDraw.Draw(pil_image)
    
    # Attempt to load a nice font, otherwise default
    try:
        # Font size is 5% of image height
        font_size = int(pil_image.height * 0.05)
        # Ensure minimum font size
        font_size = max(font_size, 20)
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    # Calculate text size using textbbox (newer Pillow) or fallback
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        # Fallback for older Pillow versions
        text_w, text_h = draw.textsize(text, font=font)
    
    # Position: Bottom Right with padding
    margin = 20
    x = pil_image.width - text_w - margin
    y = pil_image.height - text_h - margin
    
    # Draw Text with a black outline for visibility on any background
    outline_range = 2
    for offX in range(-outline_range, outline_range+1):
        for offY in range(-outline_range, outline_range+1):
            draw.text((x+offX, y+offY), text, font=font, fill="black")
            
    # Draw main white text
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 220))
    
    return pil_image

def convert_image_to_bytes(pil_img, format="PNG"):
    buf = BytesIO()
    pil_img.save(buf, format=format)
    return buf.getvalue()

# --- APP PAGES ---

def page_instructions():
    st.title("OpenCV & Streamlit Assignment")
    st.markdown("""
    ### Welcome!
    This application contains the solutions and interactive tools for the OpenCV course assignment.
    
    **How to use:**
    Use the **Sidebar** on the left to navigate between different tools:
    
    1. **E-Signature & Watermark**: A tool to sign documents or overlay watermarks transparently.
    2. **Interactive Lab**: An experiment with Logical Operations (Masking) where a custom watermark is *automatically* applied to the result.
    3. **Notebook Solutions**: Code snippets for the specific exercises (Thresholds, Logical Ops, Alpha Channel).
    """)

def page_esignature():
    st.title("Watermark & E-Signature Tool")
    st.write("Upload a document and a signature/logo to overlay them.")

    col1, col2 = st.columns(2)
    with col1:
        doc_file = st.file_uploader("1. Upload Document (Base)", type=['jpg', 'jpeg', 'png'])
    with col2:
        sign_file = st.file_uploader("2. Upload Signature/Logo (Overlay)", type=['png', 'jpg', 'jpeg'])

    if doc_file and sign_file:
        # Load images
        doc = Image.open(doc_file).convert("RGBA")
        sign = Image.open(sign_file).convert("RGBA")

        st.divider()
        st.subheader("Settings")
        
        # Sidebar-like settings in an expander or columns
        c1, c2, c3 = st.columns(3)
        with c1:
            scale = st.slider("Signature Size", 0.1, 2.0, 0.5)
        with c2:
            opacity = st.slider("Opacity", 0, 255, 220)
        with c3:
            remove_bg = st.checkbox("Remove White Background?", help="Check this if your signature is on white paper.")

        # Process Signature
        if remove_bg:
            # Convert to grayscale, invert (so ink is white, paper is black)
            gray = sign.convert("L")
            mask = ImageOps.invert(gray)
            # Use the inverted mask as the alpha channel (Ink=Opaque, White Paper=Transparent)
            sign.putalpha(mask)
            # Make the pixels black (or dark blue) for the ink
            # Create a solid color image
            ink_color = Image.new("RGBA", sign.size, (0, 0, 139, 255)) # Dark Blue
            ink_color.putalpha(mask)
            sign = ink_color

        # Resize
        new_size = (int(sign.width * scale), int(sign.height * scale))
        sign = sign.resize(new_size)

        # Opacity
        r, g, b, a = sign.split()
        a = a.point(lambda p: p * (opacity / 255))
        sign = Image.merge("RGBA", (r, g, b, a))

        # Position
        st.write("**Position Signature:**")
        x_pos = st.slider("X Position", 0, doc.width, 50)
        y_pos = st.slider("Y Position", 0, doc.height, 50)

        # Composite
        combined = doc.copy()
        combined.paste(sign, (x_pos, y_pos), sign)

        st.image(combined, caption="Final Document", use_column_width=True)

        # Download
        st.download_button(
            label="Download Signed Document",
            data=convert_image_to_bytes(combined.convert("RGB"), "JPEG"),
            file_name="signed_document.jpg",
            mime="image/jpeg"
        )

def page_interactive_lab():
    st.title("Interactive Logical Operations Lab")
    st.markdown("Upload your photo. We will apply a **Logical Mask**, and your watermark will **automatically appear** on the bottom right.")

    uploaded_file = st.file_uploader("Upload Your Photo", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        # Convert to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        st.sidebar.header("Lab Settings")
        mask_shape = st.sidebar.radio("Select Mask Shape", ["Circle", "Rectangle"])
        
        # 1. Create Mask
        mask = np.zeros(img_bgr.shape[:2], dtype="uint8")
        h, w = mask.shape
        center = (w // 2, h // 2)
        
        if mask_shape == "Circle":
            radius = min(h, w) // 3
            cv2.circle(mask, center, radius, 255, -1)
        else:
            cv2.rectangle(mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)
            
        # 2. Perform Logical Operation (Bitwise AND)
        # This keeps the image ONLY where the mask is white
        masked_img = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        
        # Convert back to PIL
        res_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
        pil_result = Image.fromarray(res_rgb)
        
        # 3. AUTO-APPLY WATERMARK (Required Feature)
        final_result = add_watermark(pil_result, text="© MY WATERMARK 2025")
        
        # Layout
        c1, c2 = st.columns(2)
        with c1:
            st.image(img_rgb, caption="Original", use_column_width=True)
            st.image(mask, caption="Generated Mask", use_column_width=True, clamp=True)
        with c2:
            st.image(final_result, caption="Result + Auto Watermark", use_column_width=True)
            
            st.success("Watermark automatically applied to bottom-right.")
            
            st.download_button(
                label="Download Result",
                data=convert_image_to_bytes(final_result, "PNG"),
                file_name="lab_result.png",
                mime="image/png"
            )

def page_notebook_solutions():
    st.title("Notebook Exercise Solutions")
    st.write("Copy and paste these blocks into your Google Colab notebooks.")
    
    st.subheader("2-2: Thresholds")
    st.code("""
import cv2
import numpy as np
# Assuming 'img' is your loaded grayscale image
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
ret2, thresh3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    """, language="python")

    st.subheader("2-3: Logical Operations")
    st.code("""
# Assuming 'img' is your image
# Create a binary mask
ret, mask = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Bitwise AND (Masking)
result = cv2.bitwise_and(img, img, mask=mask)
    """, language="python")
    
    st.subheader("2-4: Alpha Channel")
    st.code("""
img_bgr = cv2.imread('image.jpg')
b, g, r = cv2.split(img_bgr)
# Create alpha channel (255=Opaque)
alpha = np.ones_like(b) * 255
# Merge back
img_bgra = cv2.merge((b, g, r, alpha))
    """, language="python")

# --- MAIN DISPATCHER ---

def main():
    st.set_page_config(page_title="OpenCV Apps", layout="wide")
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to:", 
        ["Instructions", "E-Signature & Watermark", "Interactive Lab (Logical Ops)", "Notebook Solutions"])
        
    if app_mode == "Instructions":
        page_instructions()
    elif app_mode == "E-Signature & Watermark":
        page_esignature()
    elif app_mode == "Interactive Lab (Logical Ops)":
        page_interactive_lab()
    elif app_mode == "Notebook Solutions":
        page_notebook_solutions()

if __name__ == "__main__":
    main()
