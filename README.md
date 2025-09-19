# Object Detection App

A simple web application for object detection built with YOLO and Streamlit. You can detect objects in images, or videos.
---

## Features

- Image detection: upload images and see detected objects    
- Webcam detection: try live object detection using your webcam  
- Adjustable confidence: control the sensitivity of detections  
- Download results: save annotated images and videos  

---

## Installation

1. Clone or download this repository  
2. Install the dependencies:

```bash
pip install -r requirements.txt
streamlit run app.py

```
# Project Structure

Object Detection/
├── app.py           # Streamlit web application
├── model.py         # YOLO object detection model
├── requirements.txt # Python dependencies
└── README.md        # Project documentation
