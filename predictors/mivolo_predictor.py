"""
MiVOLO: Age and Gender Prediction

Lightweight wrapper around the MiVOLO model for face-only age/gender prediction.
Uses the original MiVOLO codebase (added to sys.path) for model creation.
"""

import os
import sys
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from typing import Dict, List, Optional, Union
from PIL import Image

DEFAULT_WEIGHT_PATH = os.environ.get(
    "MIVOLO_WEIGHTS",
    "weights/mivolo/model_utk_age_gender_4.23_97.69.pth",
)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def class_letterbox(im, new_shape=(224, 224), color=(0, 0, 0)):
    """Resize and pad image while maintaining aspect ratio."""
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    if im.shape[0] == new_shape[0] and im.shape[1] == new_shape[1]:
        return im
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im


def prepare_face_crop(img_bgr, target_size=224, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    """Prepare a single face crop for MiVOLO inference."""
    img = class_letterbox(img_bgr, new_shape=(target_size, target_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = (img - np.array(mean)) / np.array(std)
    img = img.astype(dtype=np.float32)
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)
    return torch.from_numpy(img)


class MiVOLOPredictor:
    """High-level predictor for MiVOLO age/gender model (face-only mode)."""

    def __init__(self, weight_path=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.weight_path = weight_path or DEFAULT_WEIGHT_PATH

        # Add MiVOLO repo to path for imports
        mivolo_repo = os.environ.get("MIVOLO_REPO")
        if mivolo_repo is None:
            raise RuntimeError(
                "Set MIVOLO_REPO env var to the cloned MiVOLO repo "
                "(https://github.com/WildChlamydia/MiVOLO)"
            )
        if mivolo_repo not in sys.path:
            sys.path.insert(0, mivolo_repo)

        # Load meta info from checkpoint
        state = torch.load(self.weight_path, map_location='cpu')
        self.min_age = state['min_age']
        self.max_age = state['max_age']
        self.avg_age = state['avg_age']
        self.only_age = state.get('no_gender', False)

        # Determine if this is a person+face model or face-only model
        if 'with_persons_model' in state:
            self.with_persons_model = state['with_persons_model']
        else:
            self.with_persons_model = 'patch_embed.conv1.0.weight' in state['state_dict']

        self.num_classes = 1 if self.only_age else 3
        self.in_chans = 3 if not self.with_persons_model else 6
        self.input_size = state['state_dict']['pos_embed'].shape[1] * 16

        # Create model using MiVOLO codebase
        from mivolo.model.create_timm_model import create_model
        model_name = f"mivolo_d1_{self.input_size}"
        self.model = create_model(
            model_name=model_name,
            num_classes=self.num_classes,
            in_chans=self.in_chans,
            pretrained=False,
            checkpoint_path=self.weight_path,
            filter_keys=["fds."],
        )

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_batch(self, face_crops_bgr: List[np.ndarray]) -> List[Dict]:
        """
        Predict age and gender for a batch of face crops.

        Args:
            face_crops_bgr: List of BGR numpy arrays (face crops)

        Returns:
            List of dicts: {'age': float, 'gender': str, 'gender_score': float}
        """
        if len(face_crops_bgr) == 0:
            return []

        # Prepare batch
        tensors = [prepare_face_crop(crop, self.input_size) for crop in face_crops_bgr]
        batch = torch.stack(tensors).to(self.device)

        output = self.model(batch)

        results = []
        for i in range(len(face_crops_bgr)):
            if self.only_age:
                age_raw = output[i].item()
                gender = 'unknown'
                gender_score = 0.0
            else:
                age_raw = output[i, 2].item()
                gender_output = output[i, :2].softmax(-1)
                gender_prob, gender_idx = gender_output.topk(1)
                gender = 'male' if gender_idx.item() == 0 else 'female'
                gender_score = gender_prob.item()

            age = age_raw * (self.max_age - self.min_age) + self.avg_age
            age = round(age, 2)

            results.append({
                'age': age,
                'gender': gender,
                'gender_score': round(gender_score, 4),
            })

        return results
