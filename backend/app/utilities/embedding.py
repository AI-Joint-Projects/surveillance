from transformers import ViTModel, ViTFeatureExtractor
import cv2
import torch
from torch import nn
import os
import numpy as np
from ultralytics import YOLO

# Load YOLO face detection model
yolo_model = YOLO("best.pt")

def preprocess_image(image, image_size=(224, 224)):
    """
    Preprocesses an image for the ViT model.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, image_size)  # Resize to model's input size
    image = image / 255.0  # Scale to [0, 1]
    image = (image - 0.5) / 0.5  # Normalize
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # (C, H, W)
    return image_tensor

class ViTFaceModel(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224-in21k", feat_dim=512):
        super(ViTFaceModel, self).__init__()
        self.backbone = ViTModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.backbone.config.hidden_size, feat_dim)  # Project to 512D embeddings

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0, :]  # Use CLS token as the feature
        embeddings = nn.functional.normalize(self.fc(cls_token), p=2, dim=1)  # Normalize embeddings
        return embeddings

# Load ViT model
vit_model = ViTFaceModel().to('cpu')
checkpoint = torch.load("epoch_43.pth", map_location=torch.device('cpu'))
vit_model.load_state_dict(checkpoint['model_state_dict'])
vit_model.eval()

def get_image_embedding(image_path):
    """
    Perform face detection with YOLO, preprocess the detected face, and extract embeddings.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Read and detect faces in the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Detect faces using YOLO
    detections = yolo_model(image)
    if not detections or len(detections[0].boxes) == 0:
        raise ValueError(f"No face detected in image: {image_path}")

    # Assuming the first detection is the most relevant face
    face_box = detections[0].boxes[0]  # First bounding box
    x1, y1, x2, y2 = map(int, face_box.xyxy[0].tolist())

    # Crop the detected face
    face_crop = image[y1:y2, x1:x2]

    # Preprocess the cropped face
    face_tensor = preprocess_image(face_crop)

    # Extract the embedding
    with torch.no_grad():
        embedding = vit_model(face_tensor)

    # Flatten the embedding to 1D and convert to NumPy array
    embedding_np = embedding.cpu().detach().flatten().numpy()

    return embedding_np
