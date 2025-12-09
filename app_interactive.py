import streamlit as st
from PIL import Image, ImageOps

st.title("Watermark & E-Signature App")
st.write("Upload a document and a signature/logo to overlay.")

# 1. Upload the Base Image (Document)
base_file = st.file_uploader("Upload Document (Base Image)", type=['png', 'jpg', 'jpeg'])
# 2. Upload the Watermark/Signature
watermark_file = st.file_uploader("Upload Signature/Watermark (PNG recommended)", type=['png', 'jpg', 'jpeg'])

if base_file and watermark_file:
    # Open images
    base_img = Image.open(base_file).convert("RGBA")
    watermark_img = Image.open(watermark_file).convert("RGBA")

    st.sidebar.header("Settings")
    
    # Resize Watermark
    scale = st.sidebar.slider("Signature Scale", 0.1, 2.0, 0.5)
    w_width, w_height = watermark_img.size
    new_size = (int(w_width * scale), int(w_height * scale))
    watermark_resized = watermark_img.resize(new_size)

    # Opacity Control
    opacity = st.sidebar.slider("Opacity", 0, 255, 200)
    
    # Adjust opacity of watermark
    # Get the alpha channel
    r, g, b, a = watermark_resized.split()
    # Merge with new alpha value (scaled by user input)
    a = a.point(lambda p: p * (opacity / 255))
    watermark_final = Image.merge("RGBA", (r, g, b, a))

    # Positioning
    st.sidebar.subheader("Position")
    x_pos = st.sidebar.slider("X Position", 0, base_img.width, 10)
    y_pos = st.sidebar.slider("Y Position", 0, base_img.height, 10)

    # Create Composite
    combined = base_img.copy()
    combined.paste(watermark_final, (x_pos, y_pos), watermark_final)

    st.image(combined, caption="Signed Document", use_column_width=True)

    # Download Button
    # Convert back to RGB for downloading as JPG (or keep RGBA for PNG)
    from io import BytesIO
    buf = BytesIO()
    combined.convert("RGB").save(buf, format="JPEG")
    byte_im = buf.getvalue()

    st.download_button(
        label="Download Signed Document",
        data=byte_im,
        file_name="signed_document.jpg",
        mime="image/jpeg"
    )
