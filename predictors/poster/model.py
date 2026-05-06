"""
POSTER Model for Facial Expression Recognition

Based on: POSTER: A Pyramid Cross-Fusion Transformer Network for Facial Expression Recognition (CVPR)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Union

from .ir50 import Backbone
from .mobilefacenet import MobileFaceNet
from .hyp_crossvit import HyVisionTransformer


# Weight paths — set POSTER_WEIGHTS_DIR env var to the directory containing
# affect_best.pth, mobilefacenet_model_best.pth.tar, and ir50.pth
POSTER_WEIGHT_DIR = os.environ.get("POSTER_WEIGHTS_DIR", "weights/poster")
AFFECT_7CLASS_MODEL = os.path.join(POSTER_WEIGHT_DIR, 'affect_best.pth')
MOBILEFACENET_WEIGHTS = os.path.join(POSTER_WEIGHT_DIR, 'mobilefacenet_model_best.pth.tar')
IR50_WEIGHTS = os.path.join(POSTER_WEIGHT_DIR, 'ir50.pth')

EXPRESSION_LABELS_7CLASS = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']


def load_pretrained_weights(model, checkpoint):
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    return model


class SE_block(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(input_dim, input_dim)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.relu(x1)
        x1 = self.linear2(x1)
        x1 = self.sigmod(x1)
        x = x * x1
        return x


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, target_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, target_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y_hat = self.linear(x)
        return y_hat


class POSTER(nn.Module):
    def __init__(self, img_size=224, num_classes=7, model_type="large", device='cuda'):
        super().__init__()

        if model_type == "small":
            depth = 4
        elif model_type == "base":
            depth = 6
        else:
            depth = 8

        self.img_size = img_size
        self.num_classes = num_classes
        self.device = device

        self.face_landback = MobileFaceNet([112, 112], 136)
        if os.path.exists(MOBILEFACENET_WEIGHTS):
            checkpoint = torch.load(MOBILEFACENET_WEIGHTS, map_location='cpu', weights_only=False)
            self.face_landback.load_state_dict(checkpoint['state_dict'])

        for param in self.face_landback.parameters():
            param.requires_grad = False

        self.ir_back = Backbone(50, 0.0, 'ir')
        if os.path.exists(IR50_WEIGHTS):
            checkpoint = torch.load(IR50_WEIGHTS, map_location='cpu', weights_only=False)
            self.ir_back = load_pretrained_weights(self.ir_back, checkpoint)

        self.ir_layer = nn.Linear(1024, 512)

        self.pyramid_fuse = HyVisionTransformer(
            in_chans=49, q_chanel=49, embed_dim=512,
            depth=depth, num_heads=8, mlp_ratio=2.,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1
        )

        self.se_block = SE_block(input_dim=512)
        self.head = ClassificationHead(input_dim=512, target_dim=num_classes)

    def forward(self, x):
        B_ = x.shape[0]
        x_face = F.interpolate(x, size=112)
        _, x_face = self.face_landback(x_face)
        x_face = x_face.view(B_, -1, 49).transpose(1, 2)
        x_ir = self.ir_back(x)
        x_ir = self.ir_layer(x_ir)
        y_hat = self.pyramid_fuse(x_ir, x_face)
        y_hat = self.se_block(y_hat)
        y_feat = y_hat
        out = self.head(y_hat)
        return out, y_feat

    def predict_proba(self, x):
        with torch.no_grad():
            outputs, _ = self.forward(x)
            probs = F.softmax(outputs, dim=1)
        return probs


def load_poster_model(num_classes=7, model_type='large', device='cuda',
                      checkpoint_path=None):
    model = POSTER(img_size=224, num_classes=num_classes, model_type=model_type, device=device)

    if checkpoint_path is None:
        checkpoint_path = AFFECT_7CLASS_MODEL

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        model = load_pretrained_weights(model, state_dict)

    model = model.to(device)
    model.eval()
    return model
