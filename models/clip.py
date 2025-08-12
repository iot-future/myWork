"""
CLIP (Contrastive Language-Image Pre-training) 模型实现
适用于联邦学习场景的CLIP模型

参考论文：
Learning Transferable Visual Representations with Natural Language Supervision
Radford et al., 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from core.base import BaseModel
import math


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """缩放点积注意力"""
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, v)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换并重塑为多头格式
        q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 应用注意力
        attention_output, _ = self.scaled_dot_product_attention(q, k, v, mask)
        
        # 重塑回原格式
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.w_o(attention_output)


class TransformerBlock(nn.Module):
    """Transformer块"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 多头注意力 + 残差连接
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class VisionTransformer(nn.Module):
    """视觉Transformer编码器"""
    
    def __init__(self, 
                 img_size: int = 224,
                 patch_size: int = 32,
                 in_channels: int = 3,
                 d_model: int = 512,
                 n_layers: int = 12,
                 n_heads: int = 8,
                 d_ff: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Patch嵌入
        self.patch_embed = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )
        
        # 位置编码
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_patches + 1, d_model) * 0.02
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Transformer层
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Patch嵌入
        x = self.patch_embed(x)  # (B, d_model, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, d_model)
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches+1, d_model)
        
        # 添加位置编码
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer层
        for layer in self.transformer_layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # 返回CLS token的表示
        return x[:, 0]  # (B, d_model)


class TextTransformer(nn.Module):
    """文本Transformer编码器"""
    
    def __init__(self,
                 vocab_size: int = 50000,
                 max_len: int = 77,
                 d_model: int = 512,
                 n_layers: int = 12,
                 n_heads: int = 8,
                 d_ff: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        
        self.max_len = max_len
        self.d_model = d_model
        
        # 词嵌入
        self.token_embed = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
        # Transformer层
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, text_tokens):
        seq_len = text_tokens.size(1)
        
        # 词嵌入
        x = self.token_embed(text_tokens)
        
        # 添加位置编码
        x = x + self.pos_embed[:, :seq_len, :]
        x = self.dropout(x)
        
        # Transformer层
        for layer in self.transformer_layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # 使用最后一个非padding token作为句子表示
        # 这里简化处理，使用序列的最后一个位置
        return x[:, -1]  # (B, d_model)


class CLIPModel(BaseModel):
    """CLIP模型实现
    
    包含图像编码器和文本编码器，通过对比学习训练
    """
    
    def __init__(self, 
                 img_size: int = 224,
                 patch_size: int = 32,
                 in_channels: int = 3,
                 vocab_size: int = 50000,
                 max_text_len: int = 77,
                 d_model: int = 512,
                 n_layers: int = 12,
                 n_heads: int = 8,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 temperature: float = 0.07,
                 optimizer_config: Dict[str, Any] = None):
        """
        初始化CLIP模型
        
        Args:
            img_size: 输入图像尺寸
            patch_size: 图像patch大小
            in_channels: 输入通道数
            vocab_size: 词汇表大小
            max_text_len: 最大文本长度
            d_model: 模型维度
            n_layers: Transformer层数
            n_heads: 注意力头数
            d_ff: 前馈网络维度
            dropout: Dropout比例
            temperature: 对比学习温度参数
            optimizer_config: 优化器配置
        """
        super().__init__(optimizer_config)
        
        self.temperature = temperature
        self.d_model = d_model
        
        # 图像编码器
        self.image_encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # 文本编码器
        self.text_encoder = TextTransformer(
            vocab_size=vocab_size,
            max_len=max_text_len,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # 投影层
        self.image_projection = nn.Linear(d_model, d_model)
        self.text_projection = nn.Linear(d_model, d_model)
        
        # 将所有组件组合为一个模型
        self.model = nn.ModuleDict({
            'image_encoder': self.image_encoder,
            'text_encoder': self.text_encoder,
            'image_projection': self.image_projection,
            'text_projection': self.text_projection
        })
        
        # 创建优化器
        self.create_optimizer(self.model.parameters())
        if self.optimizer is None:
            # 回退到默认AdamW（CLIP通常使用AdamW）
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
        self.criterion = self.contrastive_loss
        
    def encode_image(self, images):
        """编码图像"""
        image_features = self.image_encoder(images)
        image_features = self.image_projection(image_features)
        return F.normalize(image_features, dim=-1)
    
    def encode_text(self, text_tokens):
        """编码文本"""
        text_features = self.text_encoder(text_tokens)
        text_features = self.text_projection(text_features)
        return F.normalize(text_features, dim=-1)
    
    def forward(self, images, text_tokens):
        """前向传播"""
        image_features = self.encode_image(images)
        text_features = self.encode_text(text_tokens)
        
        return image_features, text_features
    
    def contrastive_loss(self, image_features, text_features):
        """对比损失函数"""
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
            data: 包含图像和文本的数据元组 (images, text_tokens)
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        images, text_tokens = data
        
        # 前向传播
        image_features, text_features = self.forward(images, text_tokens)
        
        # 计算损失
        loss = self.contrastive_loss(image_features, text_features)
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, data):
        """评估模型
        
        Args:
            data: 包含图像和文本的数据元组 (images, text_tokens)
        """
        self.model.eval()
        
        with torch.no_grad():
            images, text_tokens = data
            
            # 前向传播
            image_features, text_features = self.forward(images, text_tokens)
            
            # 计算损失
            loss = self.contrastive_loss(image_features, text_features)
            
            # 计算准确率（图像到文本检索）
            logits = torch.matmul(image_features, text_features.t()) / self.temperature
            predictions = torch.argmax(logits, dim=1)
            labels = torch.arange(logits.size(0), device=logits.device)
            accuracy = (predictions == labels).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }
    
    def zero_shot_classify(self, images, text_features):
        """零样本分类
        
        Args:
            images: 输入图像
            text_features: 预计算的文本特征（类别描述）
        """
        self.model.eval()
        
        with torch.no_grad():
            image_features = self.encode_image(images)
            
            # 计算相似度
            logits = torch.matmul(image_features, text_features.t()) / self.temperature
            
            return F.softmax(logits, dim=-1)
