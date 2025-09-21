"""
CLIP模型实现，支持联邦学习框架
基于Hugging Face transformers库，解耦架构设计
参考论文：Learning Transferable Visual Representations with Natural Language Supervision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer, CLIPVisionModel, CLIPTextModel
from PIL import Image
from core.base import BaseModel
from utils.device_manager import device_manager
# zero-shot 分类头

from models.classificationHead import (
    create_multi_dataset_zero_shot_classifier,
    MultiHeadImageClassifier
)

# LoRA相关导入将在下面条件导入

try:
    from lora.clip_lora import CLIPLoRAWrapper

    LORA_AVAILABLE = True
except ImportError as e:
    LORA_AVAILABLE = False
    print(f"Warning: LoRA functionality not available. Please install required dependencies: {e}")


class DeviceMixin:
    """设备管理mixin，提供设备缓存和移动功能"""

    def __init__(self):
        self._device_cache = None
        self._device_cache_dirty = True

    def _get_device(self):
        """获取模型设备 - 带缓存优化"""
        if self._device_cache is None or self._device_cache_dirty:
            try:
                self._device_cache = next(self.parameters()).device
                self._device_cache_dirty = False
            except StopIteration:
                self._device_cache = torch.device('cpu')
        return self._device_cache

    def to(self, device):
        """移动模型到指定设备并标记缓存失效"""
        result = super().to(device)
        self._device_cache_dirty = True
        return result


class BaseEncoder(torch.nn.Module, DeviceMixin):
    """编码器基类，提供公共功能"""

    def __init__(self, model_name: str, cache_dir: Optional[str] = None, device: Optional[str] = None):
        super().__init__()
        DeviceMixin.__init__(self)

        self.model_name = model_name
        self.cache_dir = cache_dir
        self.processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)

        if device:
            device_manager.move_model_to_device(self, torch.device(device))

    def save(self, filename: str, model_attr: str):
        """保存编码器"""
        print(f'Saving {self.__class__.__name__} to {filename}')
        torch.save({
            'model_state_dict': getattr(self, model_attr).state_dict(),
            'model_name': self.model_name,
            'cache_dir': self.cache_dir
        }, filename)

    @classmethod
    def load(cls, filename: str):
        """加载编码器"""
        print(f'Loading {cls.__name__} from {filename}')
        checkpoint = torch.load(filename, map_location='cpu')
        return cls(
            model_name=checkpoint['model_name'],
            cache_dir=checkpoint['cache_dir']
        )


class ImageEncoder(BaseEncoder):
    """图像编码器，基于CLIP视觉模型"""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 cache_dir: Optional[str] = None, device: Optional[str] = None):
        super().__init__(model_name, cache_dir, device)

        print(f'Loading {model_name} CLIP model.')
        self.clip_model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.feature_dim = self.clip_model.config.projection_dim  # 一般为512
        # 只保留视觉模型部分
        del self.clip_model.text_model
        del self.clip_model.text_embed_dim
        del self.clip_model.text_projection

    def forward(self, images):
        device = self._get_device()
        if isinstance(images, list) and isinstance(images[0], Image.Image):
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            pixel_values = device_manager.move_tensors_to_device(inputs['pixel_values'], device=device)
        elif isinstance(images, torch.Tensor):
            pixel_values = device_manager.move_tensors_to_device(images, device=device)
        else:
            raise ValueError("Images must be either a list of PIL Images or a torch.Tensor")

        features = self.clip_model.get_image_features(pixel_values=pixel_values)
        return features

    def save(self, filename: str):
        super().save(filename, 'vision_model')

    @classmethod
    def load(cls, filename: str):
        encoder = super().load(filename)
        checkpoint = torch.load(filename, map_location='cpu')
        encoder.vision_model.load_state_dict(checkpoint['model_state_dict'])
        return encoder


class TextEncoder(BaseEncoder):
    """文本编码器，基于CLIP文本模型"""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 cache_dir: Optional[str] = None, device: Optional[str] = None):
        super().__init__(model_name, cache_dir, device)

        print(f'Loading {model_name} text model.')
        self.text_model = CLIPTextModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.feature_dim = self.text_model.config.hidden_size

    def forward(self, inputs: Dict[str, torch.Tensor], **kwargs):
        """只支持分词后的字典输入"""
        device = self._get_device()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        text_outputs = self.text_model(**inputs)
        return text_outputs

    def save(self, filename: str):
        super().save(filename, 'text_model')

    @classmethod
    def load(cls, filename: str):
        encoder = super().load(filename)
        checkpoint = torch.load(filename, map_location='cpu')
        encoder.text_model.load_state_dict(checkpoint['model_state_dict'])
        return encoder


class ClassificationHead(torch.nn.Linear):
    """分类头，支持特征归一化"""

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
    """图像分类器，结合编码器和分类头"""

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


class FederatedCLIPModel(BaseModel, DeviceMixin):
    """联邦学习CLIP模型包装器，适配联邦学习框架"""

    def __init__(self,
                 model_name: str = "openai/clip-vit-base-patch32",
                 num_classes: int = 10,
                 normalize_features: bool = True,
                 freeze_classifier: bool = True,
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

        # 创建图像编码器
        self.image_encoder = ImageEncoder(
            model_name=self.model_name,
            cache_dir=self.cache_dir
        )

        # 构建分词器
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        # 创建文本编码器（用于构建零样本分类头）
        self.text_encoder = TextEncoder(model_name=self.model_name, cache_dir=self.cache_dir)

        # 创建多数据集零样本分类器
        self.classification_head = create_multi_dataset_zero_shot_classifier(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            dataset_names=self.dataset_names,
            device=device
        )

        # 创建多头图像分类器
        self.classifier = MultiHeadImageClassifier(self.image_encoder, self.classification_head)

        self.classifier.to(device)

        # 初始化LoRA包装器
        self.lora_wrapper = None
        self._lora_enabled = False

        # 设备缓存优化
        self._device_cache = None
        self._device_cache_dirty = True

        # 应用LoRA（如果配置中启用）
        if self.lora_config.get('enabled', False) and LORA_AVAILABLE:
            self._setup_lora()

            # LoRA启用后，重新创建优化器以包含LoRA参数和分类头参数
            if self._lora_enabled:
                lora_params = [p for p in self.image_encoder.parameters() if p.requires_grad]
                classifier_params = [p for p in self.classification_head.parameters() if p.requires_grad]
                all_trainable_params = lora_params + classifier_params
                self.create_optimizer(all_trainable_params)

        # 如果需要冻结编码器
        if freeze_classifier:
            self.classifier.freeze_head()

        # 尝试使用提供的配置创建优化器
        self.create_optimizer(self.classifier.parameters())

        # 如果提供了checkpoint路径，加载预训练权重
        if checkpoint_path is not None:
            self.load_model(checkpoint_path)

        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss()

    def _setup_lora(self):
        """设置LoRA微调"""
        if not LORA_AVAILABLE:
            print("⚠️  警告: LoRA功能不可用，请安装所需依赖")
            return

        try:
            # 创建LoRA包装器
            self.lora_wrapper = CLIPLoRAWrapper(vision_model=self.image_encoder.vision_model)

            # 简化配置处理
            vision_config = {
                'r': self.lora_config.get('r', 16),
                'lora_alpha': self.lora_config.get('lora_alpha', 32),
                'lora_dropout': self.lora_config.get('lora_dropout', 0.1),
                'target_modules': self.lora_config.get('target_modules', ["q_proj", "v_proj", "k_proj", "out_proj"])
            }

            # 应用LoRA
            self.lora_wrapper.apply_lora(vision_config=vision_config)
            self._lora_enabled = True

            # # 输出关键的LoRA统计信息
            trainable_params = self.lora_wrapper.get_trainable_parameters()
            total_original_params = sum(p.numel() for p in self.image_encoder.vision_model.parameters())

            print(
                f"🎯 LoRA设置完成 | 参数效率: {(trainable_params / total_original_params) * 100:.2f}% ({trainable_params:,}/{total_original_params:,})")

        except Exception as e:
            print(f"❌ LoRA设置失败: {e}")
            self.lora_wrapper = None
            self._lora_enabled = False

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
        if self._lora_enabled and self.lora_wrapper is not None:
            # LoRA模式：返回LoRA参数
            return self.lora_wrapper.get_lora_parameters()
        else:
            # 标准模式：返回图像编码器的可训练参数
            return {
                name: param.data.clone()
                for name, param in self.image_encoder.named_parameters()
                if param.requires_grad
            }

    def set_parameters(self, params: Dict[str, torch.Tensor]):
        """设置模型参数 - 联邦学习核心功能"""
        if self._lora_enabled and self.lora_wrapper is not None:
            # LoRA模式：设置LoRA参数
            self.lora_wrapper.set_lora_parameters(params)
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

    def train_step(self, data: torch.Tensor, labels: torch.Tensor, dataset_names: Optional[tuple] = None) -> float:
        """单步训练"""
        self.classifier.train()
        self.optimizer.zero_grad()

        device = self._get_model_device()
        data, labels = device_manager.move_tensors_to_device(data, labels, device=device)

        outputs = self.classifier(data, dataset_names[0])
        loss = self.criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.image_encoder.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def evaluate_with_dataloader(self, data_loader) -> Dict[str, float]:
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
            'lora_enabled': self._lora_enabled,
            'dataset_info': self.classification_head.get_dataset_info()
        }

        if self._lora_enabled and self.lora_wrapper is not None:
            summary.update({
                'lora_trainable_parameters': self.lora_wrapper.get_trainable_parameters(),
                'lora_status': self.lora_wrapper.is_lora_applied()
            })

        return summary

    def is_lora_enabled(self) -> bool:
        """检查是否启用了LoRA"""
        return self._lora_enabled

    def get_lora_info(self) -> Dict[str, Any]:
        """获取LoRA相关信息"""
        if not self._lora_enabled or self.lora_wrapper is None:
            return {'enabled': False}

        return {
            'enabled': True,
            'status': self.lora_wrapper.is_lora_applied(),
            'trainable_parameters': self.lora_wrapper.get_trainable_parameters(),
            'config': self.lora_config
        }

    def _compute_metrics(self, outputs: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor) -> Dict[str, float]:
        """计算评估指标的通用方法"""
        pass

    def evaluate(self, data: torch.Tensor, labels: torch.Tensor, dataset_name: str = None) -> Dict[str, float]:
        """模型评估"""
        pass

    def predict(self, data: torch.Tensor, dataset_name: str = None) -> torch.Tensor:
        """预测"""
        pass

    def predict_proba(self, data: torch.Tensor, dataset_name: str = None) -> torch.Tensor:
        """预测概率"""
        pass

    def get_features(self, data: torch.Tensor) -> torch.Tensor:
        """提取特征"""
        pass
