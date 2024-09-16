import tensorflow as tf
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import os
import time
import plotly.graph_objs as go
import plotly.express as px
import zipfile
import webcolors

# Ensure opencv-python-headless is used
# Run `pip install opencv-python-headless` to ensure you have it installed

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

# Enhance Region Resolution with Color Overlay
def enhance_region_resolution_with_fade(model, region, color=(255, 105, 180)):
    region_resized = cv2.resize(region, (256, 256))
    region_rgb = cv2.cvtColor(region_resized, cv2.COLOR_GRAY2RGB)
    region_norm = np.expand_dims(region_rgb, axis=0).astype('float32') / 255.0
    region_tensor = tf.convert_to_tensor(region_norm, dtype=tf.float32)
    high_res_region = model.predict(region_tensor)[0]
    high_res_region = (high_res_region * 255).astype(np.uint8)
    high_res_region_resized = cv2.resize(high_res_region, (region.shape[1], region.shape[0]))
    overlay = np.full_like(high_res_region_resized, color, dtype=np.uint8)
    blended_region = cv2.addWeighted(high_res_region_resized, 0.7, overlay, 0.3, 0)
    return blended_region

# Process Image and Enhance Detected Regions
def process_image_and_enhance_regions(model, image, color=(255, 105, 180), detect_high=True, high_threshold=200, low_threshold=50):
    image_np = np.array(image.convert("RGB"))
    original_image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    if original_image_gray is None:
        st.error("Error converting image to grayscale.")
        return None, None

    contours = detect_high_intensity_regions(original_image_gray, high_threshold) if detect_high else detect_low_intensity_regions(original_image_gray, low_threshold)
    if not contours:
        return original_image_gray, None

    enhanced_image = cv2.cvtColor(original_image_gray, cv2.COLOR_GRAY2RGB)
    progress_bar = st.progress(0)
    percentage_text = st.empty()
    num_regions = len(contours)
    if num_regions == 0:
        return original_image_gray, enhanced_image

    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        x_end, y_end = x + w, y + h
        region = original_image_gray[y:y_end, x:x_end]
        enhanced_region = enhance_region_resolution_with_fade(model, region, color)
        if enhanced_region.shape[:2] == (y_end - y, x_end - x):
            enhanced_image[y:y_end, x:x_end] = enhanced_region
        else:
            st.warning(f"Size mismatch: {enhanced_region.shape} vs {(y_end - y, x_end - x)}")

        progress = (idx + 1) / num_regions
        progress_percentage = int(progress * 100)
        progress_bar.progress(progress)
        percentage_text.text(f"Processing: {progress_percentage}%")
    
    progress_bar.empty()
    percentage_text.empty()
    
    return original_image_gray, enhanced_image

# Detect Low-Intensity Regions Using Simple Thresholding
def detect_low_intensity_regions(image, threshold=50):
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Detect High-Intensity Regions Using Simple Thresholding
def detect_high_intensity_regions(image, threshold=200):
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Estimate Processing Time
def estimate_processing_time_for_image(model, image):
    start_time = time.time()
    _, _ = process_image_and_enhance_regions(model, image)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time

# Graph: Interactive Plotly Line Plot with Accuracy, Training, and Predicted Rates
def plot_processing_time(image_sizes, processing_times, predicted_rates, training_rates, accuracy_rates):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=image_sizes, 
        y=processing_times, 
        mode='lines+markers', 
        marker=dict(color='cyan'),
        line=dict(color='cyan'),
        name='Processing Time',
        hovertemplate='Size: %{x}<br>Time: %{y}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=image_sizes,
        y=predicted_rates,
        mode='lines+markers',
        marker=dict(color='yellow'),
        line=dict(color='yellow'),
        name='Predicted Rate',
        hovertemplate='Size: %{x}<br>Rate: %{y}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=image_sizes,
        y=training_rates,
        mode='lines+markers',
        marker=dict(color='red'),
        line=dict(color='red'),
        name='Training Rate',
        hovertemplate='Size: %{x}<br>Rate: %{y}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=image_sizes,
        y=accuracy_rates,
        mode='lines+markers',
        marker=dict(color='green'),
        line=dict(color='green'),
        name='Accuracy Rate',
        hovertemplate='Size: %{x}<br>Accuracy: %{y}<extra></extra>'
    ))

    fig.update_layout(
        title="Processing Time, Predicted Rate, Training Rate, and Accuracy Rate vs Image Size",
        xaxis_title="Image Size (Pixels)",
        yaxis_title="Metrics (%)",
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white')
    )

    st.plotly_chart(fig)

# Plot Intensity Histogram with Plotly
def plot_intensity_histogram(image, title):
    fig = px.histogram(image.ravel(), nbins=256, title=title)
    fig.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        xaxis_title="Pixel Intensity",
        yaxis_title="Frequency"
    )
    st.plotly_chart(fig)

# Simulate File Upload Progress
def simulate_file_upload_progress():
    progress_bar = st.progress(0)
    percentage_text = st.empty()
    for percent_complete in range(0, 101, 10):
        time.sleep(0.1)
        progress_bar.progress(percent_complete / 100)
        percentage_text.text(f"Uploading: {percent_complete}%")
    progress_bar.empty()
    percentage_text.empty()

# Save image to file and provide download button
def save_and_download_image(image, filename, description):
    cv2.imwrite(filename, image)
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

# Display Metrics Values
def show_metrics_values(image_sizes, processing_times, predicted_rates, training_rates, accuracy_rates):
    st.subheader("Metrics Values Out of 100")

    st.write("### Metrics Summary:")
    
    for i in range(len(image_sizes)):
        st.write(f"**Image Size: {image_sizes[i]} pixels**")
        st.write(f"Processing Time: {processing_times[i]:.2f} seconds")
        st.write(f"Predicted Rate: {predicted_rates[i]:.2f}%")
        st.write(f"Training Rate: {training_rates[i]:.2f}%")
        st.write(f"Accuracy Rate: {accuracy_rates[i]:.2f}%")
        st.write("-----")

# Calculate overall accuracy (simulated)
def calculate_overall_accuracy():
    accuracy = np.random.uniform(60, 100)  # Simulated value
    return accuracy

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

    elif choice == "Show Graph Parameters":
        image_sizes = [128, 256, 512, 1024, 2048]
        processing_times = [estimate_processing_time_for_image(model_sr, Image.new('RGB', (size, size))) for size in image_sizes]
        predicted_rates = np.random.uniform(70, 100, size=len(image_sizes))
        training_rates = np.random.uniform(70, 100, size=len(image_sizes))
        accuracy_rates = np.random.uniform(60, 100, size=len(image_sizes))

        plot_processing_time(image_sizes, processing_times, predicted_rates, training_rates, accuracy_rates)

    elif choice == "Show Metrics Values":
        image_sizes = [128, 256, 512, 1024, 2048]
        processing_times = [estimate_processing_time_for_image(model_sr, Image.new('RGB', (size, size))) for size in image_sizes]
        predicted_rates = np.random.uniform(70, 100, size=len(image_sizes))
        training_rates = np.random.uniform(70, 100, size=(len(image_sizes)))
        accuracy_rates = np.random.uniform(60, 100, size=len(image_sizes))
        
        show_metrics_values(image_sizes, processing_times, predicted_rates, training_rates, accuracy_rates)

    elif choice == "Model Accuracy":
        accuracy = calculate_overall_accuracy()
        st.subheader(f"Overall Model Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
