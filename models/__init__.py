"""
模型模块

包含联邦学习支持的各种深度学习模型：
- CNNModel: 适用于MNIST的卷积神经网络
- SimpleLinearModel: 简单线性模型  
- CLIPModel: 完整版CLIP模型（视觉-语言多模态）
"""

from .base import SimpleLinearModel
from .cnn import CNNModel
from .clip import CLIPModel

__all__ = [
    'SimpleLinearModel',
    'CNNModel', 
    'CLIPModel'
]