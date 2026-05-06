"""
L2CS-Net: Gaze estimation model.

Self-contained module extracted from the L2CS-Net repository.
Uses ResNet50 backbone with 90-bin classification for yaw and pitch.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from typing import Dict, List, Tuple, Union


DEFAULT_WEIGHT_PATH = os.environ.get(
    "L2CS_WEIGHTS",
    "weights/l2cs/L2CSNet_gaze360.pkl",
)
NUM_BINS = 90


class L2CS(nn.Module):
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(L2CS, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_yaw_gaze = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch_gaze = nn.Linear(512 * block.expansion, num_bins)
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        pre_yaw_gaze = self.fc_yaw_gaze(x)
        pre_pitch_gaze = self.fc_pitch_gaze(x)
        return pre_yaw_gaze, pre_pitch_gaze


def getArch(arch, bins):
    if arch == 'ResNet18':
        model = L2CS(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], bins)
    else:
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], bins)
    return model


def softmax_temperature(tensor, temperature):
    result = torch.exp(tensor / temperature)
    result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))
    return result


class L2CSPredictor:
    """High-level wrapper for L2CS-Net gaze prediction."""

    def __init__(self, weight_path=None, arch='ResNet50', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.weight_path = weight_path or DEFAULT_WEIGHT_PATH
        self.num_bins = NUM_BINS

        self.model = getArch(arch, self.num_bins)

        if os.path.exists(self.weight_path):
            state_dict = torch.load(self.weight_path, map_location='cpu', weights_only=False)
            # Handle 'module.' prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            load_result = self.model.load_state_dict(new_state_dict, strict=False)
            if load_result.missing_keys:
                print(f"[L2CS] Warning: missing keys in checkpoint: {load_result.missing_keys}", flush=True)
            if load_result.unexpected_keys:
                print(f"[L2CS] Warning: unexpected keys in checkpoint: {load_result.unexpected_keys}", flush=True)

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Create idx tensor for bin-to-angle conversion
        self.idx_tensor = torch.arange(self.num_bins, dtype=torch.float32).to(self.device)

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
        return self.transform(image)

    @torch.no_grad()
    def predict_batch(self, face_crops: List) -> List[Dict]:
        """
        Predict gaze for a batch of face crops.

        Returns:
            List of dicts with 'yaw' and 'pitch' in radians.
        """
        if len(face_crops) == 0:
            return []

        tensors = [self.preprocess(img) for img in face_crops]
        batch = torch.stack(tensors).to(self.device)

        yaw_logits, pitch_logits = self.model(batch)

        yaw_probs = F.softmax(yaw_logits, dim=1)
        pitch_probs = F.softmax(pitch_logits, dim=1)

        # Bin to angle: each bin = 4 degrees, range [-180, 180], 90 bins
        yaw_predicted = torch.sum(yaw_probs * self.idx_tensor, dim=1) * 4 - 180
        pitch_predicted = torch.sum(pitch_probs * self.idx_tensor, dim=1) * 4 - 180

        # Convert to radians
        yaw_rad = yaw_predicted * math.pi / 180.0
        pitch_rad = pitch_predicted * math.pi / 180.0

        results = []
        for i in range(len(face_crops)):
            results.append({
                'yaw': float(yaw_rad[i].cpu()),
                'pitch': float(pitch_rad[i].cpu()),
            })
        return results

    @torch.no_grad()
    def predict_with_confidence(self, face_crop) -> Dict:
        """
        Predict gaze for a single face crop, including a confidence score.

        Confidence = min(max(yaw_softmax), max(pitch_softmax)).
        With 90 bins the uniform baseline is ~0.011; a confident prediction
        will have its peak bin well above this (typically 0.15–0.50 for a
        frontal face; near baseline for closed/occluded eyes).

        Args:
            face_crop: BGR numpy array or PIL Image.

        Returns:
            dict with keys 'yaw' (rad), 'pitch' (rad), 'confidence' (float).
        """
        tensor = self.preprocess(face_crop).unsqueeze(0).to(self.device)
        yaw_logits, pitch_logits = self.model(tensor)

        yaw_probs = F.softmax(yaw_logits, dim=1)
        pitch_probs = F.softmax(pitch_logits, dim=1)

        yaw_predicted = torch.sum(yaw_probs * self.idx_tensor, dim=1) * 4 - 180
        pitch_predicted = torch.sum(pitch_probs * self.idx_tensor, dim=1) * 4 - 180

        yaw_rad = float((yaw_predicted * math.pi / 180.0)[0].cpu())
        pitch_rad = float((pitch_predicted * math.pi / 180.0)[0].cpu())

        yaw_conf = float(torch.max(yaw_probs).cpu())
        pitch_conf = float(torch.max(pitch_probs).cpu())

        return {
            'yaw': yaw_rad,
            'pitch': pitch_rad,
            'confidence': min(yaw_conf, pitch_conf),
        }

    @torch.no_grad()
    def predict_batch_with_confidence(self, face_crops: List) -> List[Dict]:
        """
        Predict gaze for a batch of face crops, including per-sample confidence.

        Confidence = min(max(yaw_softmax), max(pitch_softmax)).

        Returns:
            List of dicts with 'yaw' (rad), 'pitch' (rad), 'confidence' (float).
        """
        if not face_crops:
            return []

        tensors = [self.preprocess(img) for img in face_crops]
        batch   = torch.stack(tensors).to(self.device)

        yaw_logits, pitch_logits = self.model(batch)

        yaw_probs   = F.softmax(yaw_logits,   dim=1)
        pitch_probs = F.softmax(pitch_logits, dim=1)

        yaw_predicted   = torch.sum(yaw_probs   * self.idx_tensor, dim=1) * 4 - 180
        pitch_predicted = torch.sum(pitch_probs * self.idx_tensor, dim=1) * 4 - 180
        yaw_rad   = yaw_predicted   * math.pi / 180.0
        pitch_rad = pitch_predicted * math.pi / 180.0

        yaw_confs   = torch.max(yaw_probs,   dim=1)[0]
        pitch_confs = torch.max(pitch_probs, dim=1)[0]

        results = []
        for i in range(len(face_crops)):
            results.append({
                'yaw':        float(yaw_rad[i].cpu()),
                'pitch':      float(pitch_rad[i].cpu()),
                'confidence': min(float(yaw_confs[i].cpu()), float(pitch_confs[i].cpu())),
            })
        return results
