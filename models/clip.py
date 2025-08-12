"""
CLIP (Contrastive Language-Image Pre-training) 模型实现
基于 Hugging Face transformers 的精简实现，适用于联邦学习场景

参考论文：
Learning Transferable Visual Representations with Natural Language Supervision
Radford et al., 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, Union
from core.base import BaseModel
from transformers import (
    CLIPModel as HFCLIPModel, # Hugging Face 的完整 CLIP 模型实现
    CLIPProcessor, # CLIP 模型的预处理器
    CLIPConfig, # CLIP 模型的配置类
    CLIPVisionModel,
    CLIPTextModel
)
from PIL import Image
import warnings


class CLIPModel(BaseModel):
    """基于 Hugging Face 的 CLIP 模型实现
    
    使用预训练的 CLIP 模型，支持联邦学习场景下的参数聚合和本地训练
    """
    
    def __init__(self, 
                 model_name: str = "openai/clip-vit-base-patch32",
                 temperature: float = 0.07,
                 freeze_vision_encoder: bool = False,
                 freeze_text_encoder: bool = False,
                 optimizer_config: Dict[str, Any] = None,
                 use_pretrained: bool = True):
        """
        初始化CLIP模型
        
        Args:
            model_name: 预训练模型名称 (默认: "openai/clip-vit-base-patch32")
            temperature: 对比学习温度参数
            freeze_vision_encoder:W 是否冻结视觉编码器
            freeze_text_encoder: 是否冻结文本编码器  
            optimizer_config: 优化器配置
            use_pretrained: 是否使用预训练权重
        """
        super().__init__(optimizer_config)
        
        self.model_name = model_name
        self.temperature = temperature
        self.freeze_vision_encoder = freeze_vision_encoder
        self.freeze_text_encoder = freeze_text_encoder
        
        # 初始化处理器和模型
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        if use_pretrained:
            # 加载预训练模型
            self.model = HFCLIPModel.from_pretrained(model_name)
        else:
            # 使用配置创建随机初始化的模型
            config = CLIPConfig.from_pretrained(model_name)
            self.model = HFCLIPModel(config)
        
        # 冻结指定的编码器
        if freeze_vision_encoder:
            for param in self.model.vision_model.parameters():
                param.requires_grad = False
                
        if freeze_text_encoder:
            for param in self.model.text_model.parameters():
                param.requires_grad = False
        
        # 创建优化器
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.create_optimizer(trainable_params)
        if self.optimizer is None:
            # 回退到默认AdamW
            self.optimizer = torch.optim.AdamW(trainable_params, lr=5e-5, weight_decay=0.1)
        
        self.criterion = self.contrastive_loss
        
    def encode_image(self, images: Union[torch.Tensor, list]):
        """编码图像
        
        Args:
            images: 图像张量或PIL图像列表
            
        Returns:
            归一化的图像特征
        """
        if isinstance(images, list):
            # 处理PIL图像列表
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            image_features = self.model.get_image_features(**inputs)
        else:
            # 处理张量输入
            image_features = self.model.get_image_features(pixel_values=images)
        
        return F.normalize(image_features, dim=-1)
    
    def encode_text(self, texts: Union[torch.Tensor, list]):
        """编码文本
        
        Args:
            texts: 文本token张量或字符串列表
            
        Returns:
            归一化的文本特征
        """
        if isinstance(texts, list):
            # 处理字符串列表
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            text_features = self.model.get_text_features(**inputs)
        else:
            # 处理张量输入
            text_features = self.model.get_text_features(input_ids=texts)
        
        return F.normalize(text_features, dim=-1)
    
    def forward(self, images, texts):
        """前向传播"""
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)
        
        return image_features, text_features
    
    def contrastive_loss(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """对比损失函数
        
        Args:
            image_features: 归一化的图像特征
            text_features: 归一化的文本特征
            
        Returns:
            对比损失值
        """
        # 计算相似度矩阵
        logits = torch.matmul(image_features, text_features.t()) / self.temperature
        
        batch_size = logits.size(0)
        labels = torch.arange(batch_size, device=logits.device)
        
        # 图像到文本的交叉熵损失
        loss_i2t = F.cross_entropy(logits, labels)
        
        # 文本到图像的交叉熵损失
        loss_t2i = F.cross_entropy(logits.t(), labels)
        
        return (loss_i2t + loss_t2i) / 2
    
    def get_parameters(self) -> Dict[str, Any]:
        """获取模型参数"""
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
    
    def set_parameters(self, params: Dict[str, Any]):
        """设置模型参数"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params:
                    param.data.copy_(params[name])
    
    def train_step(self, data):
        """单步训练
        
        Args:
            data: 包含图像和文本的数据，格式为 (images, texts)
                 images: PIL图像列表或图像张量
                 texts: 文本字符串列表或token张量
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        images, texts = data
        
        try:
            # 前向传播
            image_features, text_features = self.forward(images, texts)
            
            # 计算损失
            loss = self.contrastive_loss(image_features, text_features)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            return loss.item()
            
        except Exception as e:
            print(f"Training step error: {e}")
            return float('inf')
    
    def evaluate(self, data):
        """评估模型
        
        Args:
            data: 包含图像和文本的数据，格式为 (images, texts)
        """
        self.model.eval()
        
        with torch.no_grad():
            images, texts = data
            
            try:
                # 前向传播
                image_features, text_features = self.forward(images, texts)
                
                # 计算损失
                loss = self.contrastive_loss(image_features, text_features)
                
                # 计算准确率（图像到文本检索）
                logits = torch.matmul(image_features, text_features.t()) / self.temperature
                predictions = torch.argmax(logits, dim=1)
                labels = torch.arange(logits.size(0), device=logits.device)
                accuracy = (predictions == labels).float().mean()
                
                # 计算召回率@k
                top5_acc = self._calculate_recall_at_k(logits, k=5)
                
                return {
                    'loss': loss.item(),
                    'accuracy': accuracy.item(),
                    'recall@5': top5_acc
                }
                
            except Exception as e:
                print(f"Evaluation error: {e}")
                return {
                    'loss': float('inf'),
                    'accuracy': 0.0,
                    'recall@5': 0.0
                }
    
    def _calculate_recall_at_k(self, logits: torch.Tensor, k: int = 5) -> float:
        """计算 Recall@K 指标"""
        batch_size = logits.size(0)
        if k >= batch_size:
            return 1.0
            
        # 获取top-k预测
        _, topk_indices = torch.topk(logits, k, dim=1)
        
        # 创建正确标签
        labels = torch.arange(batch_size, device=logits.device).unsqueeze(1)
        
        # 计算命中率
        hits = (topk_indices == labels).any(dim=1).float()
        return hits.mean().item()
    
    def zero_shot_classify(self, images, class_texts: list):
        """零样本分类
        
        Args:
            images: 输入图像（PIL列表或张量）
            class_texts: 类别文本描述列表
            
        Returns:
            分类概率分布
        """
        self.model.eval()
        
        with torch.no_grad():
            # 编码图像
            image_features = self.encode_image(images)
            
            # 编码类别文本
            text_features = self.encode_text(class_texts)
            
            # 计算相似度
            logits = torch.matmul(image_features, text_features.t()) / self.temperature
            
            return F.softmax(logits, dim=-1)
    
    def get_text_embeddings(self, texts: list) -> torch.Tensor:
        """获取文本嵌入（用于缓存类别描述）"""
        return self.encode_text(texts)
    
    def get_image_embeddings(self, images) -> torch.Tensor:
        """获取图像嵌入"""
        return self.encode_image(images)
    
    def compute_similarity(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """计算图像和文本特征之间的相似度"""
        return torch.matmul(image_features, text_features.t()) / self.temperature
    
    def save_pretrained(self, save_directory: str):
        """保存模型和处理器"""
        self.model.save_pretrained(save_directory)
        self.processor.save_pretrained(save_directory)
    
    def load_pretrained(self, load_directory: str):
        """加载模型和处理器"""
        self.model = HFCLIPModel.from_pretrained(load_directory)
        self.processor = CLIPProcessor.from_pretrained(load_directory)
