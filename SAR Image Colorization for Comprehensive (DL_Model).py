import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageOps
import os
import time
import plotly.express as px
import zipfile
import webcolors

# Load Pretrained Model from .h5 File
def load_pretrained_model(model_path):
    with st.spinner('Loading pretrained model...'):
        if os.path.isfile(model_path):
            try:
                model = tf.keras.models.load_model(model_path)
                st.success(f"Successfully loaded model from {model_path}")
            except Exception as e:
                st.error(f"Error loading model: {e}")
                model = None
        else:
            st.error(f"File not found: {model_path}")
            model = None
    return model

# Enhance Region Resolution with Color Overlay using PIL
def enhance_region_resolution_with_fade(model, region, color=(255, 105, 180)):
    # Resize the region to match the model input size (assuming 256x256)
    region_resized = region.resize((256, 256))
    # Convert to RGB for further processing
    region_rgb = region_resized.convert("RGB")
    # Normalize the pixel values
    region_np = np.array(region_rgb).astype('float32') / 255.0
    region_tensor = np.expand_dims(region_np, axis=0)
    # Pass the region through the model to get the enhanced version
    high_res_region = model.predict(region_tensor)[0]
    high_res_region = (high_res_region * 255).astype(np.uint8)
    # Convert back to PIL image
    high_res_image = Image.fromarray(high_res_region)
    high_res_image = high_res_image.resize(region.size)
    # Create a color overlay and blend with the high-res image
    overlay = Image.new("RGB", region.size, color)
    blended_region = Image.blend(high_res_image, overlay, alpha=0.3)
    return blended_region

# Process Image and Enhance Detected Regions using PIL
def process_image_and_enhance_regions(model, image, color=(255, 105, 180), detect_high=True, high_threshold=200, low_threshold=50):
    # Convert the image to grayscale
    original_image_gray = image.convert("L")
    
    if original_image_gray is None:
        st.error("Error converting image to grayscale.")
        return None, None

    # Detect regions based on intensity thresholding
    contours = detect_high_intensity_regions(original_image_gray, high_threshold) if detect_high else detect_low_intensity_regions(original_image_gray, low_threshold)
    if not contours:
        return original_image_gray, None

    # Create a colorized version of the grayscale image (to start with)
    enhanced_image = ImageOps.colorize(original_image_gray, black="black", white="white")
    progress_bar = st.progress(0)
    percentage_text = st.empty()
    num_regions = len(contours)
    if num_regions == 0:
        return original_image_gray, enhanced_image

    # Process each detected region and enhance
    for idx, contour in enumerate(contours):
        x, y, w, h = contour
        # Crop the region of interest from the original grayscale image
        region = original_image_gray.crop((x, y, x + w, y + h))
        # Enhance the region and paste it back onto the colorized image
        enhanced_region = enhance_region_resolution_with_fade(model, region, color)
        enhanced_image.paste(enhanced_region, (x, y))

        # Update progress
        progress = (idx + 1) / num_regions
        progress_percentage = int(progress * 100)
        progress_bar.progress(progress)
        percentage_text.text(f"Processing: {progress_percentage}%")

    progress_bar.empty()
    percentage_text.empty()

    return original_image_gray, enhanced_image

# Detect Low-Intensity Regions Using PIL
def detect_low_intensity_regions(image, threshold=50):
    # Convert the image to a numpy array for processing
    image_np = np.array(image)
    # Create a mask where pixels below the threshold are marked
    mask = image_np < threshold
    contours = []
    for y, row in enumerate(mask):
        for x, pixel in enumerate(row):
            if pixel:
                contours.append((x, y, 1, 1))  # Dummy contours as (x, y, w, h)
    return contours

# Detect High-Intensity Regions Using PIL
def detect_high_intensity_regions(image, threshold=200):
    # Convert the image to a numpy array for processing
    image_np = np.array(image)
    # Create a mask where pixels above the threshold are marked
    mask = image_np > threshold
    contours = []
    for y, row in enumerate(mask):
        for x, pixel in enumerate(row):
            if pixel:
                contours.append((x, y, 1, 1))  # Dummy contours as (x, y, w, h)
    return contours

# Plot Intensity Histogram with Plotly
def plot_intensity_histogram(image, title):
    # Convert the image to a numpy array for plotting
    image_np = np.array(image)
    # Plot the histogram using Plotly
    fig = px.histogram(image_np.ravel(), nbins=256, title=title)
    fig.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        xaxis_title="Pixel Intensity",
        yaxis_title="Frequency"
    )
    st.plotly_chart(fig)

# Save image to file and provide download button
def save_and_download_image(image, filename, description):
    # Save the image to disk
    image.save(filename)
    # Provide a download button for the saved image
    with open(filename, "rb") as file:
        st.download_button(label=f"Download {description}", data=file, file_name=filename)

# Save images to a ZIP file for bulk download
def save_images_to_zip(original_image_gray, enhanced_image, filenames):
    zip_filename = "processed_images.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        zipf.write(filenames[0])
        zipf.write(filenames[1])
    with open(zip_filename, "rb") as file:
        st.download_button(label="Download Processed Images (ZIP)", data=file, file_name=zip_filename)

# Streamlit UI
def main():
    st.title("SAR Image Colorization and Super-Resolution")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    options = ["Image Upload and Processing", "Show Graph Parameters", "Show Metrics Values", "Model Accuracy"]
    choice = st.sidebar.radio("Go to", options)

    # Load model
    model_path = r'super_resolution_model.h5'
    model_sr = load_pretrained_model(model_path)
    
    if model_sr is None:
        st.error("Model not loaded. Please check the model file.")
        return

    if choice == "Image Upload and Processing":
        uploaded_file = st.file_uploader("Upload SAR Image", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)
            except Exception as e:
                st.error(f"Failed to load image: {e}")
                return
            
            color = st.color_picker("Pick a Color for Enhancement", '#FF69B4')  # Default is hot pink
            color_rgb = webcolors.hex_to_rgb(color)

            detect_high = st.sidebar.checkbox("Detect High Intensity Regions", value=True)
            high_threshold = st.sidebar.slider("Set High-Intensity Threshold", 0, 255, 200)
            low_threshold = st.sidebar.slider("Set Low-Intensity Threshold", 0, 255, 50)

            if st.button("Process Image"):
                original_image_gray, enhanced_image = process_image_and_enhance_regions(
                    model_sr, image, color_rgb, detect_high, high_threshold, low_threshold
                )

                if original_image_gray is not None and enhanced_image is not None:
                    st.image(enhanced_image, caption='Enhanced Image', use_column_width=True)

                    save_and_download_image(original_image_gray, "original_image.jpg", "Original Image")
                    save_and_download_image(enhanced_image, "enhanced_image.jpg", "Enhanced Image")
                    save_images_to_zip("original_image.jpg", "enhanced_image.jpg", ["original_image.jpg", "enhanced_image.jpg"])

                    st.subheader("Intensity Histogram")
                    plot_intensity_histogram(original_image_gray, "Original Image Histogram")
                    plot_intensity_histogram(enhanced_image, "Enhanced Image Histogram")

if __name__ == "__main__":
    main()
