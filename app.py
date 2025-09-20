import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import os

# Handle OpenCV import for Streamlit Cloud
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError as e:
    st.error(f"OpenCV import failed: {e}")
    CV2_AVAILABLE = False

if CV2_AVAILABLE:
    from model import ObjectDetector


def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Object Detection",
        page_icon="🔍",
        layout="centered"
    )
    
    # Title
    st.title("🔍 Object Detection")
    
    # Check if OpenCV is available
    if not CV2_AVAILABLE:
        st.error("OpenCV is not available. Please check the deployment configuration.")
        st.stop()
    
    # Initialize detector
    @st.cache_resource
    def load_detector():
        return ObjectDetector()
    
    detector = load_detector()
    
    # Confidence slider
    confidence = st.slider("Confidence", 0.1, 1.0, 0.4, 0.1)
    detector.set_confidence_threshold(confidence)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload image or video",
        type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov']
    )
    
    # Process uploaded file
    if uploaded_file is not None:
        file_type = uploaded_file.type.split('/')[0]
        
        if file_type == 'image':
            process_image(uploaded_file, detector)
        elif file_type == 'video':
            process_video(uploaded_file, detector)


def process_image(uploaded_file, detector):
    """Process uploaded image and display results."""
    
    # Convert uploaded file to image
    image = Image.open(uploaded_file)
    image_array = np.array(image)
    
    # Convert PIL image to OpenCV format
    if len(image_array.shape) == 3:
        image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    else:
        image_cv = image_array
    
    # Save temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        cv2.imwrite(tmp_file.name, image_cv)
        temp_path = tmp_file.name
    
    try:
        # Run detection
        with st.spinner("Detecting..."):
            annotated_image, detections = detector.detect_image(temp_path)
        
        # Display results
        annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, use_container_width=True)
        
        # Show detection count
        if detections:
            st.write(f"Found {len(detections)} objects")
        else:
            st.write("No objects detected")
        
        # Download button
        st.download_button(
            "Download",
            cv2.imencode('.jpg', annotated_image)[1].tobytes(),
            "detected_image.jpg",
            "image/jpeg"
        )
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def process_video(uploaded_file, detector):
    """Process uploaded video and display results."""
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name
    
    try:
        # Run detection
        with st.spinner("Processing..."):
            output_path = detector.detect_video(temp_path)
        
        # Display video
        st.video(output_path)
        
        # Download button
        with open(output_path, 'rb') as f:
            st.download_button(
                "Download",
                f.read(),
                "detected_video.mp4",
                "video/mp4"
            )
    
    finally:
        # Clean up temporary files
        for path in [temp_path, output_path]:
            if os.path.exists(path):
                os.unlink(path)


if __name__ == "__main__":
    main()
