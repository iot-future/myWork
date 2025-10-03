"""
CLIP模型实现，支持联邦学习框架
基于Hugging Face transformers库，解耦架构设计
参考论文：Learning Transferable Visual Representations with Natural Language Supervision
"""
import os
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List
from transformers import CLIPTokenizer
from PIL import Image
from core.base import BaseModel
from utils.device_manager import device_manager, DeviceMixin
from models.encoder import ImageEncoder, TextEncoder
from models.classificationHead import (
    create_multi_dataset_zero_shot_classifier,
    MultiHeadImageClassifier
)

try:
    import lora.loRA_utils as lora_utils

    LORA_AVAILABLE = True
except ImportError as e:
    LORA_AVAILABLE = False
    print(f"Warning: LoRA functionality not available. Please install required dependencies: {e}")


class FederatedCLIPModel(BaseModel, DeviceMixin):
    """联邦学习CLIP模型包装器，适配联邦学习框架"""

    def __init__(self,
                 model_name: str = "openai/clip-vit-base-patch32",
                 num_classes: int = 10,
                 normalize_features: bool = True,
                 freeze_classification_head: bool = True,
                 cache_dir: Optional[str] = None,
                 optimizer_config: Optional[Dict[str, Any]] = None,
                 checkpoint_path: Optional[str] = None,
                 lora_config: Optional[Dict[str, Any]] = None,
                 dataset_names: Optional[List[str]] = None,
                 device: Optional[Union[str, torch.device]] = None):
        """初始化联邦学习CLIP模型"""
        super().__init__(optimizer_config)
        DeviceMixin.__init__(self)

        self.model_name = model_name
        self.num_classes = num_classes
        self.normalize_features = normalize_features
        self.cache_dir = cache_dir
        self.lora_config = lora_config or {}
        self.dataset_names = dataset_names
        self.is_lora_applied = False

        # 创建图像编码器
        self.image_encoder = ImageEncoder(
            model_name=self.model_name,
            cache_dir=self.cache_dir
        )

        # 构建分词器
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir, use_fast=True)
        # 创建文本编码器（用于构建零样本分类头）
        self.text_encoder = TextEncoder(model_name=self.model_name, cache_dir=self.cache_dir)

        # 创建多数据集零样本分类器
        self.classification_head = create_multi_dataset_zero_shot_classifier(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            dataset_names=self.dataset_names,
            device=device,
            head_dir=os.path.join(self.cache_dir, "zero_shot_heads")
        )

        # 创建多头图像分类器
        self.classifier = MultiHeadImageClassifier(self.image_encoder, self.classification_head)

        self.classifier.to(device)

        # 初始化LoRA包装器
        self.classifier.lora_model = None

        # 设备缓存优化
        self._device_cache = None
        self._device_cache_dirty = True

        # 如果需要冻结编码器
        if freeze_classification_head:
            self.classifier.freeze_head()

        # 应用LoRA（如果配置中启用）
        if self.lora_config.get('enabled', False) and LORA_AVAILABLE:
            self._setup_lora()

        # 如果提供了checkpoint路径，加载预训练权重
        if checkpoint_path is not None:
            self.load_model(checkpoint_path)

        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 尝试使用提供的配置创建优化器
        self.create_optimizer(self.classifier.parameters())

    def _setup_lora(self):
        """设置LoRA微调"""
        if not LORA_AVAILABLE:
            print("⚠️  警告: LoRA功能不可用，请安装所需依赖")
            return

        try:
            # 简化配置处理
            image_config = {
                'r': self.lora_config.get('r', 16),
                'lora_alpha': self.lora_config.get('lora_alpha', 32),
                'lora_dropout': self.lora_config.get('lora_dropout', 0.1),
                'target_modules': self.lora_config.get('target_modules', ["q_proj", "v_proj", "k_proj", "out_proj"])
            }

            # 应用LoRA
            self.classifier.image_encoder.clip_model = lora_utils.apply_lora(
                model=self.classifier.image_encoder.clip_model, **image_config)
            self.is_lora_applied = True
            self.classifier.is_lora_applied = True

        except Exception as e:
            print(f"❌ LoRA设置失败: {e}")
            self.classifier.image_encoder = self.image_encoder

    def to(self, device):
        """将模型移动到指定设备"""
        device_manager.move_model_to_device(self.image_encoder, device)
        device_manager.move_model_to_device(self.text_encoder, device)
        device_manager.move_model_to_device(self.classification_head, device)
        device_manager.move_model_to_device(self.criterion, device)
        # 设备发生变化时，标记缓存失效
        self._device_cache_dirty = True
        return self

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """获取模型参数 - 联邦学习核心功能"""
        if self.is_lora_applied and self.classifier.lora_model is not None:
            # LoRA模式：返回LoRA参数
            return lora_utils.get_lora_parameters(self.classifier)
        else:
            # 标准模式：返回图像编码器的可训练参数
            return {
                name: param.data.clone()
                for name, param in self.image_encoder.named_parameters()
                if param.requires_grad
            }

    def set_parameters(self, params: Dict[str, torch.Tensor]):
        """设置模型参数 - 联邦学习核心功能"""
        params = deepcopy(params)  # 深拷贝参数
        if self.is_lora_applied and self.classifier.lora_model is not None:
            # LoRA模式：设置LoRA参数
            lora_utils.set_lora_parameters(self.classifier, params)
        else:
            # 标准模式：设置图像编码器参数
            with torch.no_grad():
                for name, param in self.image_encoder.named_parameters():
                    if name in params and param.requires_grad:
                        param.data.copy_(params[name])

    def _get_model_device(self):
        """获取模型所在设备 - 带缓存优化"""
        if self._device_cache is None or self._device_cache_dirty:
            try:
                self._device_cache = next(self.image_encoder.parameters()).device
                self._device_cache_dirty = False
            except StopIteration:
                self._device_cache = torch.device('cpu')
        return self._device_cache

    def train_step(self, data: torch.Tensor, labels: torch.Tensor, dataset_names: Optional[tuple] = None):
        """单步训练"""
        if self.is_lora_applied:
            self.classifier.image_encoder.train()
        else:
            self.image_encoder.train()
        self.optimizer.zero_grad()

        outputs = self.classifier(inputs=data, dataset_name=dataset_names[0])

        loss = self.criterion(outputs, labels)
        loss.backward()

        self.optimizer.step()

        return loss.item()

    def evaluate(self, data_loader) -> Dict[str, float]:
        """使用数据加载器评估模型"""
        self.classifier.eval()
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        top5_correct = 0
        device = self._get_model_device()

        with torch.no_grad():
            for batch_data, batch_labels, dataset_names in data_loader:
                batch_data, batch_labels = device_manager.move_tensors_to_device(
                    batch_data, batch_labels, device=device
                )

                outputs = self.classifier(batch_data, dataset_names[0])
                loss = self.criterion(outputs, batch_labels)

                batch_size = batch_data.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == batch_labels).sum().item()

                if self.num_classes >= 5:
                    _, top5_pred = outputs.topk(5, 1, largest=True, sorted=True)
                    top5_correct += top5_pred.eq(batch_labels.view(-1, 1).expand_as(top5_pred)).sum().item()

        result = {
            'loss': total_loss / total_samples if total_samples > 0 else 0.0,
            'accuracy': correct_predictions / total_samples if total_samples > 0 else 0.0
        }

        if self.num_classes >= 5:
            result['top5_accuracy'] = top5_correct / total_samples if total_samples > 0 else 0.0

        return result

    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'image_encoder_state_dict': self.image_encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'model_name': self.model_name,
                'num_classes': self.num_classes,
                'normalize_features': self.normalize_features,
                'cache_dir': self.cache_dir,
                'dataset_name': self.dataset_name
            }
        }, filepath)
        print(f"CLIP model saved to {filepath}")

    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.image_encoder.load_state_dict(checkpoint['image_encoder_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer state loaded")
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}")
        print(f"CLIP model loaded from {filepath}")

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, **kwargs):
        """从checkpoint文件创建CLIP模型"""
        print(f"Creating CLIP model from checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint.get('model_config', {})

        init_kwargs = {
            'model_name': config.get('model_name', 'openai/clip-vit-base-patch32'),
            'num_classes': config.get('num_classes', 10),
            'normalize_features': config.get('normalize_features', True),
            'cache_dir': config.get('cache_dir', None),
            'checkpoint_path': checkpoint_path
        }
        init_kwargs.update(kwargs)

        return cls(**init_kwargs)

    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要"""
        total_params = sum(p.numel() for p in self.image_encoder.parameters())
        trainable_params = sum(p.numel() for p in self.image_encoder.parameters() if p.requires_grad)

        summary = {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'encoder_feature_dim': self.image_encoder.feature_dim,
            'normalize_features': self.normalize_features,
            'is_lora_applied': self.is_lora_applied,
            'dataset_info': self.classification_head.get_dataset_info()
        }

        if self.is_lora_applied and self.classifier.lora_model is not None:
            summary.update({
                'lora_trainable_parameters': lora_utils.get_trainable_parameters(self.classifier),
                'lora_status': lora_utils.is_lora_applied(self.classifier)
            })

        return summary

    def is_lora_applied(self) -> bool:
        """检查是否启用了LoRA"""
        return self.is_lora_applied

    def get_lora_info(self) -> Dict[str, Any]:
        """获取LoRA相关信息"""
        if not self.is_lora_applied or self.classifier.lora_model is None:
            return {'enabled': False}

        return {
            'enabled': True,
            'status': lora_utils.is_lora_applied(self.classifier),
            'trainable_parameters': lora_utils.get_trainable_parameters(self.classifier),
            'config': self.lora_config
        }

    # def train(self, mode=True):
    #     """切换训练/评估模式"""
    #     self.classifier.train(mode)


class ClassificationHead(torch.nn.Linear):
    """分类头，支持特征归一化（未使用）"""

    def __init__(self, input_size: int, output_size: int, normalize: bool = False, bias: bool = True):
        super().__init__(input_size, output_size, bias=bias)
        self.normalize = normalize
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            inputs = F.normalize(inputs, dim=-1, p=2)
        return super().forward(inputs)

    def save(self, filename: str):
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
    """图像分类器，结合编码器和分类头（未使用）"""

    def __init__(self, image_encoder: ImageEncoder, classification_head: ClassificationHead):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head

    def freeze_encoder(self):
        for param in self.image_encoder.parameters():
            param.requires_grad_(False)

    def unfreeze_encoder(self):
        for param in self.image_encoder.parameters():
            param.requires_grad_(True)

    def forward(self, inputs):
        features = self.image_encoder(inputs)
        return self.classification_head(features)

    def save(self, filename: str):
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
        print(f'Loading image classifier from {filename}')
        checkpoint = torch.load(filename, map_location='cpu')

        image_encoder = ImageEncoder(model_name=checkpoint['encoder_model_name'])
        image_encoder.load_state_dict(checkpoint['image_encoder'])

        head_config = checkpoint['head_config']
        classification_head = ClassificationHead(**head_config)
        classification_head.load_state_dict(checkpoint['classification_head'])

        return cls(image_encoder, classification_head)
