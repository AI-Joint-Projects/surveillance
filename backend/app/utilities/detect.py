"""
This module  is solely dedicated for object detection codes, particularly:
1. Person Detection 
2. Face Detection
"""
from ultralytics import YOLO
import cv2

face_model = YOLO('best.pt') 
person_model= YOLO('yolov8n.pt')

def detect_face(image):
    results = face_model(image)
    boxes = []  # List to store bounding box coordinates
    for result in results:
        xyxy_boxes = result.boxes.xyxy.cpu().numpy() 
        for box in xyxy_boxes:
            x1, y1, x2, y2 = map(int, box[:4])  # Extract and convert coordinates
            boxes.append((x1, y1, x2, y2))  # Append as a tuple
    return boxes

def detect_person(image):
    """
    Detects person in an image using YOLOv8.
    
    Args:
        image (numpy.ndarray): The input image to detect persons in.

    Returns:
        results (ultralytics.results.Results): The detection results.
    """
    results = person_model.predict(source=image, conf=0.5, classes=[0])
    return results