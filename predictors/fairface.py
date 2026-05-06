"""
FairFace: Face Attribute Prediction for Ethnicity

Adapted from toolbox for use in the face annotation pipeline.
Provides ethnicity prediction using ResNet34 backbone.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms
from typing import Dict, List, Union


DEFAULT_MODEL_PATH = os.environ.get(
    "FAIRFACE_WEIGHTS",
    "weights/fairface/res34_fair_align_multi_7_20190809.pt",
)

RACE_LABELS_7 = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
GENDER_LABELS = ['Male', 'Female']
AGE_LABELS = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
AGE_GROUP_CENTERS = [1, 6, 14.5, 24.5, 34.5, 44.5, 54.5, 64.5, 75]


class FairFaceModel(nn.Module):
    def __init__(self, num_classes=18, pretrained_backbone=False):
        super().__init__()
        self.backbone = models.resnet34(pretrained=pretrained_backbone)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

    def predict(self, x):
        outputs = self.forward(x)
        race_logits = outputs[:, :7]
        gender_logits = outputs[:, 7:9]
        age_logits = outputs[:, 9:18]
        race_prob = torch.softmax(race_logits, dim=1)
        gender_prob = torch.softmax(gender_logits, dim=1)
        age_prob = torch.softmax(age_logits, dim=1)
        return {
            'race_pred': torch.argmax(race_prob, dim=1),
            'race_prob': race_prob,
            'gender_pred': torch.argmax(gender_prob, dim=1),
            'gender_prob': gender_prob,
            'age_pred': torch.argmax(age_prob, dim=1),
            'age_prob': age_prob,
        }


class FairFacePredictor:
    def __init__(self, model_path=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.model = FairFaceModel(num_classes=18, pretrained_backbone=False)

        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
            new_state_dict = {f"backbone.{k}": v for k, v in state_dict.items()}
            self.model.load_state_dict(new_state_dict)

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess(self, image):
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = Image.fromarray(image[:, :, ::-1])  # BGR to RGB
            else:
                image = Image.fromarray(image)
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be numpy array or PIL Image")
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return self.transform(image).unsqueeze(0)

    @torch.no_grad()
    def predict_batch(self, images: List[Union[np.ndarray, Image.Image]]) -> List[Dict]:
        if len(images) == 0:
            return []
        tensors = [self.preprocess(img) for img in images]
        batch = torch.cat(tensors, dim=0).to(self.device)
        preds = self.model.predict(batch)
        results = []
        for i in range(len(images)):
            race_idx = preds['race_pred'][i].item()
            results.append({
                'race': RACE_LABELS_7[race_idx],
                'race_idx': race_idx,
                'race_prob': preds['race_prob'][i].cpu().numpy(),
            })
        return results
