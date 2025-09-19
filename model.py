import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional
import tempfile
import os
import subprocess


class ObjectDetector:
    """A simple and clean object detection class using YOLO."""
    
    def __init__(self, model_path: str = 'yolo11s.pt'):
        """
        Initialize the object detector with YOLO model.
        
        Args:
            model_path: Path to the YOLO model file
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = 0.4
    
    def _get_colors(self, cls_num: int) -> Tuple[int, int, int]:
        """
        Generate colors for different object classes.
        
        Args:
            cls_num: Class number
            
        Returns:
            RGB color tuple
        """
        base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        color_index = cls_num % len(base_colors)
        increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
        color = [base_colors[color_index][i] + increments[color_index][i] * 
                (cls_num // len(base_colors)) % 256 for i in range(3)]
        return tuple(color)
    
    def _draw_detections(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame.
        
        Args:
            frame: Input image frame
            results: YOLO detection results
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for result in results:
            classes_names = result.names
            
            for box in result.boxes:
                if box.conf[0] > self.confidence_threshold:
                    # Get coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Get class info
                    cls = int(box.cls[0])
                    class_name = classes_names[cls]
                    confidence = box.conf[0]
                    
                    # Get color
                    color = self._get_colors(cls)
                    
                    # Draw rectangle (normal thickness)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw label (normal size)
                    label = f'{class_name} {confidence:.2f}'
                    cv2.putText(annotated_frame, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return annotated_frame
    
    def detect_image(self, image_path: str) -> Tuple[np.ndarray, List[dict]]:
        """
        Detect objects in a single image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple of (annotated_image, detection_info)
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Run detection
        results = self.model(image)
        
        # Draw detections
        annotated_image = self._draw_detections(image, results)
        
        # Extract detection info
        detection_info = []
        for result in results:
            for box in result.boxes:
                if box.conf[0] > self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    class_name = result.names[cls]
                    confidence = float(box.conf[0])
                    
                    detection_info.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2]
                    })
        
        return annotated_image, detection_info
    
    def detect_video(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        Detect objects in a video file.
        
        Args:
            video_path: Path to the input video
            output_path: Path to save the output video (optional)
            
        Returns:
            Path to the output video
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video from {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Reduce resolution for faster processing
        new_width = min(width, 640)
        new_height = int(height * new_width / width)
        
        # Set up output video with browser-compatible codec
        if output_path is None:
            output_path = tempfile.mktemp(suffix='_detected.mp4')
        
        # Try multiple codecs for Streamlit Cloud compatibility
        codecs = ['mp4v', 'XVID', 'MJPG']
        out = None
        
        for codec in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
            if out.isOpened():
                break
            else:
                out.release()
        
        if not out or not out.isOpened():
            raise RuntimeError("Could not initialize video writer with any codec")
        
        try:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames for faster processing (process every 2nd frame)
                frame_count += 1
                if frame_count % 2 != 0:
                    continue
                
                # Resize frame for faster processing
                frame = cv2.resize(frame, (new_width, new_height))
                
                # Run detection
                results = self.model(frame)
                
                # Draw detections
                annotated_frame = self._draw_detections(frame, results)
                
                # Write frame
                out.write(annotated_frame)
        
        finally:
            cap.release()
            out.release()
        
        # Convert to web-compatible format
        web_compatible_path = self._convert_to_web_format(output_path)
        
        # Clean up original if different from web-compatible
        if web_compatible_path != output_path and os.path.exists(output_path):
            os.unlink(output_path)
        
        return web_compatible_path
    
        """
        Detect objects from webcam for a specified duration.
        
        Args:
            duration: Duration in seconds
            
        Returns:
            Path to the output video
        """
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Could not open webcam")
        
        # Get webcam properties
        fps = 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set up output video
        output_path = tempfile.mktemp(suffix='_webcam_detected.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        max_frames = duration * fps
        
        try:
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection
                results = self.model(frame)
                
                # Draw detections
                annotated_frame = self._draw_detections(frame, results)
                
                # Write frame
                out.write(annotated_frame)
                frame_count += 1
        
        finally:
            cap.release()
            out.release()
        
        return output_path
    
    def _convert_to_web_format(self, input_path: str) -> str:
        """Convert video to web-compatible format using ffmpeg."""
        try:
            output_path = tempfile.mktemp(suffix='_web_compatible.mp4')
            cmd = [
                'ffmpeg', '-i', input_path,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-movflags', '+faststart',
                '-y',  # Overwrite output file
                output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            # If ffmpeg fails, return original file
            return input_path
    
    def set_confidence_threshold(self, threshold: float):
        """
        Set the confidence threshold for detections.
        
        Args:
            threshold: Confidence threshold (0.0 to 1.0)
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
