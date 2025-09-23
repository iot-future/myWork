"""
模型模块

包含联邦学习支持的各种深度学习模型：
- FederatedCNNModel: 适用于MNIST的卷积神经网络
- FederatedLinearModel: 联邦学习线性模型  
- FederatedCLIPModel: 完整版CLIP模型（视觉-语言多模态）
- ZeroShotClassificationHead: 基于文本模板的零样本分类头
- ZeroShotClassifier: 零样本分类器
"""

from .base import FederatedLinearModel
from .cnn import FederatedCNNModel
from .clip import FederatedCLIPModel
from .zero_shot_classifier import (
    ZeroShotClassificationHead,
    ZeroShotClassifier,
    CIFAR10ZeroShotHead,
    CIFAR100ZeroShotHead,
    MNISTZeroShotHead,
    create_zero_shot_head
)

__all__ = [
    "FederatedLinearModel",
    "FederatedCNNModel", 
    "FederatedCLIPModel",
    "ZeroShotClassificationHead",
    "ZeroShotClassifier",
    "CIFAR10ZeroShotHead",
    "CIFAR100ZeroShotHead", 
    "MNISTZeroShotHead",
    "ImageNetZeroShotHead",
    "create_zero_shot_head"
]
