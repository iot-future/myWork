"""
CLIP (Contrastive Language-Image Pre-training) 模型实现
基于Hugging Face transformers库的实现，支持联邦学习框架
解耦架构设计，分离图像编码器、文本编码器和分类头

参考论文：
Learning Transferable Visual Representations with Natural Language Supervision
Radford et al., 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List, Tuple
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel, CLIPTextModel
from transformers import AutoProcessor, AutoModel
from PIL import Image
from core.base import BaseModel


class ImageEncoder(torch.nn.Module):
    """
    图像编码器类
    基于Hugging Face CLIP模型的图像编码器，用于将图像转换为特征向量
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", 
                 cache_dir: Optional[str] = None, device: Optional[str] = None):
        """
        初始化图像编码器
        
        Args:
            model_name: 预训练模型名称，默认为"openai/clip-vit-base-patch32"
            cache_dir: 模型缓存目录
            device: 设备类型
        """
        super().__init__()
        
        print(f'Loading {model_name} pre-trained weights.')
        
        # 使用Hugging Face的CLIP视觉模型
        self.vision_model = CLIPVisionModel.from_pretrained(
            model_name, 
            cache_dir=cache_dir
        )
        
        # 创建处理器用于图像预处理
        self.processor = CLIPProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        self.cache_dir = cache_dir
        self.model_name = model_name
        
        # 获取特征维度
        self.feature_dim = self.vision_model.config.hidden_size
        
        if device:
            self.to(device)

    def forward(self, images):
        """
        前向传播，将图像编码为特征向量
        
        Args:
            images: 输入的图像张量或PIL图像列表
            
        Returns:
            编码后的图像特征向量
        """
        # 如果输入是PIL图像列表，先进行预处理
        if isinstance(images, list) and isinstance(images[0], Image.Image):
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            pixel_values = inputs['pixel_values'].to(self.vision_model.device)
        elif isinstance(images, torch.Tensor):
            pixel_values = images
        else:
            raise ValueError("Images must be either a list of PIL Images or a torch.Tensor")
            
        # 通过视觉编码器获取特征
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        # 返回pooled输出（CLS token的表示）
        return vision_outputs.pooler_output
    

    def save(self, filename: str):
        """
        保存图像编码器到checkpoint文件
        
        Args:
            filename: 保存文件的路径
        """
        print(f'Saving image encoder to {filename}')
        torch.save({
            'model_state_dict': self.vision_model.state_dict(),
            'model_name': self.model_name,
            'cache_dir': self.cache_dir
        }, filename)

    @classmethod
    def load(cls, filename: str):
        """
        从checkpoint加载图像编码器
        
        Args:
            filename: 加载checkpoin文件的路径
            
        Returns:
            加载的图像编码器实例
        """
        print(f'Loading image encoder from {filename}')
        checkpoint = torch.load(filename, map_location='cpu')
        
        encoder = cls(
            model_name=checkpoint['model_name'],
            cache_dir=checkpoint['cache_dir']
        )
        encoder.vision_model.load_state_dict(checkpoint['model_state_dict'])
        return encoder


class TextEncoder(torch.nn.Module):
    """
    文本编码器类
    基于Hugging Face CLIP模型的文本编码器，用于将文本转换为特征向量
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 cache_dir: Optional[str] = None, device: Optional[str] = None):
        """
        初始化文本编码器
        
        Args:
            model_name: 预训练模型名称
            cache_dir: 模型缓存目录
            device: 设备类型
        """
        super().__init__()
        
        print(f'Loading {model_name} text encoder pre-trained weights.')
        
        # 使用Hugging Face的CLIP文本模型
        self.text_model = CLIPTextModel.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        # 创建处理器用于文本预处理
        self.processor = CLIPProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        self.cache_dir = cache_dir
        self.model_name = model_name
        
        # 获取特征维度
        self.feature_dim = self.text_model.config.hidden_size
        
        if device:
            self.to(device)

    def forward(self, texts: Union[List[str], torch.Tensor]):
        """
        前向传播，将文本编码为特征向量
        
        Args:
            texts: 输入的文本列表或token张量
            
        Returns:
            编码后的文本特征向量
        """
        if isinstance(texts, list):
            # 文本预处理
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs['input_ids'].to(self.text_model.device)
            attention_mask = inputs['attention_mask'].to(self.text_model.device)
        elif isinstance(texts, torch.Tensor):
            input_ids = texts
            attention_mask = None
        else:
            raise ValueError("Texts must be either a list of strings or a torch.Tensor")
            
        # 通过文本编码器获取特征
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        # 返回pooled输出
        return text_outputs.pooler_output
    

    def save(self, filename: str):
        """保存文本编码器到checkpoint文件"""
        print(f'Saving text encoder to {filename}')
        torch.save({
            'model_state_dict': self.text_model.state_dict(),
            'model_name': self.model_name,
            'cache_dir': self.cache_dir
        }, filename)

    @classmethod
    def load(cls, filename: str):
        """加载从checkpoint文件的文本编码器"""
        print(f'Loading text encoder from {filename}')
        checkpoint = torch.load(filename, map_location='cpu')
        
        encoder = cls(
            model_name=checkpoint['model_name'],
            cache_dir=checkpoint['cache_dir']
        )
        encoder.text_model.load_state_dict(checkpoint['model_state_dict'])
        return encoder


class ClassificationHead(torch.nn.Linear):
    """
    分类头类
    继承自torch.nn.Linear，用于将特征向量映射到类别概率
    支持特征归一化功能
    """
    def __init__(self, input_size: int, output_size: int, normalize: bool = False, 
                 bias: bool = True):
        """
        初始化分类头
        
        Args:
            input_size: 输入特征维度
            output_size: 输出类别数
            normalize: 是否对输入特征进行L2归一化
            bias: 是否使用偏置
        """
        super().__init__(input_size, output_size, bias=bias)
        self.normalize = normalize
        
        # 初始化权重
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            inputs: 输入特征向量
            
        Returns:
            分类logits
        """
        # 如果需要归一化，对输入进行L2归一化
        if self.normalize:
            inputs = F.normalize(inputs, dim=-1, p=2)
        return super().forward(inputs)

    def __call__(self, inputs):
        """使对象可调用"""
        return self.forward(inputs)

    def save(self, filename: str):
        """保存分类头"""
        print(f'Saving classification head to {filename}')
        torch.save({
            'state_dict': self.state_dict(),
            'input_size': self.in_features,
            'output_size': self.out_features,
            'normalize': self.normalize,
            'bias': self.bias is not None
        }, filename)

    @classmethod
    def load(cls, filename: str):
        """加载分类头"""
        print(f'Loading classification head from {filename}')
        checkpoint = torch.load(filename, map_location='cpu')
        
        head = cls(
            input_size=checkpoint['input_size'],
            output_size=checkpoint['output_size'],
            normalize=checkpoint['normalize'],
            bias=checkpoint['bias']
        )
        head.load_state_dict(checkpoint['state_dict'])
        return head


class ImageClassifier(torch.nn.Module):
    """
    图像分类器类
    结合图像编码器和分类头的完整图像分类模型
    """
    def __init__(self, image_encoder: ImageEncoder, classification_head: ClassificationHead):
        """
        初始化图像分类器
        
        Args:
            image_encoder: 图像编码器实例
            classification_head: 分类头实例
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head

    def freeze_encoder(self):
        """冻结图像编码器的参数，使其在训练时不更新"""
        for param in self.image_encoder.parameters():
            param.requires_grad_(False)

    def unfreeze_encoder(self):
        """解冻图像编码器的参数"""
        for param in self.image_encoder.parameters():
            param.requires_grad_(True)

    def freeze_head(self):
        """冻结分类头的参数，使其在训练时不更新"""
        for param in self.classification_head.parameters():
            param.requires_grad_(False)

    def unfreeze_head(self):
        """解冻分类头的参数"""
        for param in self.classification_head.parameters():
            param.requires_grad_(True)

    def forward(self, inputs):
        """
        前向传播
        
        Args:
            inputs: 输入图像
            
        Returns:
            分类结果
        """
        # 通过图像编码器提取特征
        features = self.image_encoder(inputs)
        # 通过分类头得到分类结果
        outputs = self.classification_head(features)
        return outputs

    def __call__(self, inputs):
        """使对象可调用"""
        return self.forward(inputs)

    def save(self, filename: str):
        """保存图像分类器"""
        print(f'Saving image classifier to {filename}')
        torch.save({
            'image_encoder': self.image_encoder.state_dict(),
            'classification_head': self.classification_head.state_dict(),
            'encoder_model_name': self.image_encoder.model_name,
            'head_config': {
                'input_size': self.classification_head.in_features,
                'output_size': self.classification_head.out_features,
                'normalize': self.classification_head.normalize,
                'bias': self.classification_head.bias is not None
            }
        }, filename)

    @classmethod
    def load(cls, filename: str):
        """加载图像分类器"""
        print(f'Loading image classifier from {filename}')
        checkpoint = torch.load(filename, map_location='cpu')
        
        # 重建图像编码器
        image_encoder = ImageEncoder(model_name=checkpoint['encoder_model_name'])
        image_encoder.load_state_dict(checkpoint['image_encoder'])
        
        # 重建分类头
        head_config = checkpoint['head_config']
        classification_head = ClassificationHead(
            input_size=head_config['input_size'],
            output_size=head_config['output_size'],
            normalize=head_config['normalize'],
            bias=head_config['bias']
        )
        classification_head.load_state_dict(checkpoint['classification_head'])
        
        return cls(image_encoder, classification_head)


class CLIPModel(BaseModel):
    """
    完整的CLIP模型实现
    继承自BaseModel，适配联邦学习框架
    支持图像分类任务
    """
    def __init__(self, 
                 model_name: str = "openai/clip-vit-base-patch32",
                 num_classes: int = 10,
                 normalize_features: bool = True,
                 freeze_encoder: bool = False,
                 cache_dir: Optional[str] = None,
                 optimizer_config: Optional[Dict[str, Any]] = None,
                 checkpoint_path: Optional[str] = None):
        """
        初始化CLIP模型
        
        Args:
            model_name: 预训练模型名称
            num_classes: 分类类别数
            normalize_features: 是否对特征进行归一化
            freeze_encoder: 是否冻结编码器
            cache_dir: 模型缓存目录
            optimizer_config: 优化器配置
            checkpoint_path: 如果提供，将从此路径加载预训练权重
        """
        # 调用父类构造函数
        super().__init__(optimizer_config)
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.normalize_features = normalize_features
        self.cache_dir = cache_dir
        
        # 创建图像编码器
        self.image_encoder = ImageEncoder(
            model_name=self.model_name,
            cache_dir=self.cache_dir
        )
        
        # 创建分类头
        self.classification_head = ClassificationHead(
            input_size=self.image_encoder.feature_dim,
            output_size=self.num_classes,
            normalize=self.normalize_features
        )
        
        # 组合成完整的分类器
        self.classifier = ImageClassifier(self.image_encoder, self.classification_head)
        
        # 如果需要冻结编码器
        if freeze_encoder:
            self.classifier.freeze_encoder()
        
        # 创建AdamW优化器
        self.create_optimizer(self.classifier.parameters())
        if self.optimizer is None:
            # 回退到默认AdamW配置
            from utils.optimizer_factory import OptimizerFactory
            # CLIP专用的默认配置
            default_config = {
                'learning_rate': 5e-5,
                'weight_decay': 0.1,
                'betas': [0.9, 0.98],
                'eps': 1e-6
            }
            self.optimizer = OptimizerFactory.create_optimizer(
                self.classifier.parameters(), default_config
            )
        
        # 如果提供了checkpoint路径，加载预训练权重
        if checkpoint_path is not None:
            self.load_model(checkpoint_path)
        
        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss()
        
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        获取模型参数 - 联邦学习核心功能
        
        Returns:
            参数名称到参数张量的映射
        """
        return {
            name: param.data.clone()
            for name, param in self.classifier.named_parameters()
            if param.requires_grad
        }
    
    def set_parameters(self, params: Dict[str, torch.Tensor]):
        """
        设置模型参数 - 联邦学习核心功能
        
        Args:
            params: 参数名称到参数张量的映射
        """
        with torch.no_grad():
            for name, param in self.classifier.named_parameters():
                if name in params and param.requires_grad:
                    param.data.copy_(params[name])
    
    def train_step(self, data: torch.Tensor, labels: torch.Tensor) -> float:
        """
        单步训练
        
        Args:
            data: 输入图像数据
            labels: 标签
            
        Returns:
            训练损失
        """
        self.classifier.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        outputs = self.classifier(data)
        loss = self.criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪（可选）
        torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, data: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        模型评估
        
        Args:
            data: 评估数据
            labels: 真实标签
            
        Returns:
            评估指标字典
        """
        self.classifier.eval()
        
        with torch.no_grad():
            outputs = self.classifier(data)
            loss = self.criterion(outputs, labels)
            
            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            accuracy = correct / total
            
            # 计算Top-5准确率（如果类别数>=5）
            top5_accuracy = None
            if self.num_classes >= 5:
                _, top5_pred = outputs.topk(5, 1, largest=True, sorted=True)
                top5_correct = top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
                top5_accuracy = top5_correct / total
        
        result = {
            'loss': loss.item(),
            'accuracy': accuracy
        }
        
        if top5_accuracy is not None:
            result['top5_accuracy'] = top5_accuracy
            
        return result
    
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """
        预测
        
        Args:
            data: 输入数据
            
        Returns:
            预测结果
        """
        self.classifier.eval()
        with torch.no_grad():
            outputs = self.classifier(data)
            _, predicted = torch.max(outputs, 1)
        return predicted
    
    def predict_proba(self, data: torch.Tensor) -> torch.Tensor:
        """
        预测概率
        
        Args:
            data: 输入数据
            
        Returns:
            预测概率
        """
        self.classifier.eval()
        with torch.no_grad():
            outputs = self.classifier(data)
            probabilities = F.softmax(outputs, dim=1)
        return probabilities
    
    def get_features(self, data: torch.Tensor) -> torch.Tensor:
        """
        提取特征
        
        Args:
            data: 输入数据
            
        Returns:
            特征向量
        """
        self.classifier.eval()
        with torch.no_grad():
            features = self.image_encoder(data)
        return features
    
    def save_model(self, filepath: str):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        torch.save({
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'model_name': self.model_name,
                'num_classes': self.num_classes,
                'normalize_features': self.normalize_features,
                'cache_dir': self.cache_dir
            }
        }, filepath)
        print(f"CLIP model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer state loaded")
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}")
        print(f"CLIP model loaded from {filepath}")
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, **kwargs):
        """
        从checkpoint文件创建CLIP模型的类方法
        
        Args:
            checkpoint_path: checkpoint文件路径
            **kwargs: 额外的初始化参数，将覆盖checkpoint中的配置
            
        Returns:
            从checkpoint加载的CLIP模型实例
        """
        print(f"Creating CLIP model from checkpoint: {checkpoint_path}")
        
        # 加载checkpoint获取配置
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint.get('model_config', {})
        
        # 合并checkpoint配置和传入的参数，传入的参数具有更高优先级
        init_kwargs = {
            'model_name': config.get('model_name', 'openai/clip-vit-base-patch32'),
            'num_classes': config.get('num_classes', 10),
            'normalize_features': config.get('normalize_features', True),
            'cache_dir': config.get('cache_dir', None),
            'checkpoint_path': checkpoint_path  # 自动加载权重
        }
        
        # 用传入的参数覆盖checkpoint配置
        init_kwargs.update(kwargs)
        
        return cls(**init_kwargs)
    
    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要"""
        total_params = sum(p.numel() for p in self.classifier.parameters())
        trainable_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'encoder_feature_dim': self.image_encoder.feature_dim,
            'normalize_features': self.normalize_features
        }


# 统一的工厂函数，支持从配置或checkpoint创建CLIP模型
def create_clip_model(config: Dict[str, Any]) -> CLIPModel:
    """
    创建CLIP模型的统一工厂函数
    
    Args:
        config: 模型配置字典，可以包含以下键：
            - model_name: 预训练模型名称
            - num_classes: 分类类别数
            - normalize_features: 是否对特征进行归一化
            - freeze_encoder: 是否冻结编码器
            - cache_dir: 模型缓存目录
            - optimizer_config: 优化器配置
            - checkpoint_path: 如果提供，将从此路径加载模型权重
        
    Returns:
        CLIP模型实例
    """
    # 如果提供了checkpoint路径，优先使用from_checkpoint方法
    if 'checkpoint_path' in config and config['checkpoint_path'] is not None:
        checkpoint_path = config.pop('checkpoint_path')
        return CLIPModel.from_checkpoint(checkpoint_path, **config)
    
    # 否则直接创建新模型
    return CLIPModel(
        model_name=config.get('model_name', 'openai/clip-vit-base-patch32'),
        num_classes=config.get('num_classes', 10),
        normalize_features=config.get('normalize_features', True),
        freeze_encoder=config.get('freeze_encoder', False),
        cache_dir=config.get('cache_dir', None),
        optimizer_config=config.get('optimizer_config', None)
    )
