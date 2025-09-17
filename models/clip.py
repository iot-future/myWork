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
from utils.device_manager import device_manager

# LoRA相关导入
try:
    from lora.clip_lora import CLIPLoRAWrapper
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False
    print("Warning: LoRA functionality not available. Please install required dependencies.")


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
        
        # 设备缓存优化
        self._device_cache = None
        self._device_cache_dirty = True
        
        # 设置设备
        if device:
            device_manager.move_model_to_device(self, torch.device(device))

    def _get_device(self):
        """获取模型设备 - 带缓存优化"""
        if self._device_cache is None or self._device_cache_dirty:
            try:
                self._device_cache = next(self.vision_model.parameters()).device
                self._device_cache_dirty = False
            except StopIteration:
                self._device_cache = torch.device('cpu')
        return self._device_cache

    def to(self, device):
        """移动模型到指定设备并标记缓存失效"""
        result = super().to(device)
        self._device_cache_dirty = True
        return result

    def forward(self, images):
        """
        前向传播，将图像编码为特征向量
        
        Args:
            images: 输入的图像张量或PIL图像列表
            
        Returns:
            编码后的图像特征向量
        """
        # 获取设备（带缓存优化）
        device = self._get_device()
        
        # 如果输入是PIL图像列表，先进行预处理
        if isinstance(images, list) and isinstance(images[0], Image.Image):
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            pixel_values = device_manager.move_tensors_to_device(inputs['pixel_values'], device=device)
        elif isinstance(images, torch.Tensor):
            pixel_values = device_manager.move_tensors_to_device(images, device=device)
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
        
        # 设备缓存优化
        self._device_cache = None
        self._device_cache_dirty = True
        
        if device:
            device_manager.move_model_to_device(self, torch.device(device))

    def _get_device(self):
        """获取模型设备 - 带缓存优化"""
        if self._device_cache is None or self._device_cache_dirty:
            try:
                self._device_cache = next(self.text_model.parameters()).device
                self._device_cache_dirty = False
            except StopIteration:
                self._device_cache = torch.device('cpu')
        return self._device_cache

    def to(self, device):
        """移动模型到指定设备并标记缓存失效"""
        result = super().to(device)
        self._device_cache_dirty = True
        return result

    def forward(self, texts: Union[List[str], torch.Tensor]):
        """
        前向传播，将文本编码为特征向量
        
        Args:
            texts: 输入的文本列表或token张量
            
        Returns:
            编码后的文本特征向量
        """
        # 获取设备（带缓存优化）
        device = self._get_device()
        
        if isinstance(texts, list):
            # 文本预处理
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            input_ids, attention_mask = device_manager.move_tensors_to_device(
                inputs['input_ids'], inputs['attention_mask'], device=device
            )
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


class FederatedCLIPModel(BaseModel):
    """联邦学习CLIP模型包装器
    
    这是一个包装器类，将CLIP多模态模型适配到联邦学习框架中。
    提供统一的参数管理、设备兼容、训练接口等联邦学习特性。
    
    完整的CLIP模型实现，继承自BaseModel，适配联邦学习框架，支持图像分类任务。
    """
    def __init__(self, 
                 model_name: str = "openai/clip-vit-base-patch32",
                 num_classes: int = 10,
                 normalize_features: bool = True,
                 freeze_encoder: bool = False,
                 cache_dir: Optional[str] = None,
                 optimizer_config: Optional[Dict[str, Any]] = None,
                 checkpoint_path: Optional[str] = None,
                 lora_config: Optional[Dict[str, Any]] = None):
        """
        初始化联邦学习CLIP模型包装器
        
        Args:
            model_name: 预训练模型名称
            num_classes: 分类类别数
            normalize_features: 是否对特征进行归一化
            freeze_encoder: 是否冻结编码器
            cache_dir: 模型缓存目录
            optimizer_config: 优化器配置
            checkpoint_path: 如果提供，将从此路径加载预训练权重
            lora_config: LoRA配置，包含enabled、r、lora_alpha等参数
        """
        # 调用父类构造函数
        super().__init__(optimizer_config)
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.normalize_features = normalize_features
        self.cache_dir = cache_dir
        self.lora_config = lora_config or {}
        
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
        
        # 初始化LoRA包装器
        self.lora_wrapper = None
        self._lora_enabled = False
        
        # 设备缓存优化
        self._device_cache = None
        self._device_cache_dirty = True
        
        # 应用LoRA（如果配置中启用）
        if self.lora_config.get('enabled', False) and LORA_AVAILABLE:
            self._setup_lora()
        
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
            
            # 输出关键的LoRA统计信息
            trainable_params = self.lora_wrapper.get_trainable_parameters()
            total_original_params = sum(p.numel() for p in self.image_encoder.vision_model.parameters())
            
            print(f"🎯 LoRA设置完成 | 参数效率: {(trainable_params/total_original_params)*100:.2f}% ({trainable_params:,}/{total_original_params:,})")
                
        except Exception as e:
            print(f"❌ LoRA设置失败: {e}")
            self.lora_wrapper = None
            self._lora_enabled = False
        
    def to(self, device):
        """将模型移动到指定设备"""
        device_manager.move_model_to_device(self.classifier, device)
        device_manager.move_model_to_device(self.criterion, device)
        # 设备发生变化时，标记缓存失效
        self._device_cache_dirty = True
        return self
        
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        获取模型参数 - 联邦学习核心功能
        
        当启用LoRA时，只返回LoRA参数和分类头参数
        当未启用LoRA时，返回所有可训练参数
        
        Returns:
            参数名称到参数张量的映射
        """
        if self._lora_enabled and self.lora_wrapper is not None:
            # 获取LoRA参数
            lora_params = self.lora_wrapper.get_lora_parameters()
            
            # 获取分类头参数
            classifier_params = {
                f"classifier.{name}": param.data.clone()
                for name, param in self.classification_head.named_parameters()
                if param.requires_grad
            }
            
            # 合并LoRA参数和分类头参数
            all_params = {**lora_params, **classifier_params}
            return all_params
        else:
            # 标准模式：返回所有可训练参数
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
        if self._lora_enabled and self.lora_wrapper is not None:
            # 分离LoRA参数和分类头参数
            lora_params = {}
            classifier_params = {}
            
            for name, param in params.items():
                if name.startswith("vision.") or name.startswith("text."):
                    lora_params[name] = param
                elif name.startswith("classifier."):
                    classifier_params[name[11:]] = param  # 移除"classifier."前缀
            
            # 设置LoRA参数
            if lora_params:
                self.lora_wrapper.set_lora_parameters(lora_params)
            
            # 设置分类头参数
            if classifier_params:
                with torch.no_grad():
                    for name, param in self.classification_head.named_parameters():
                        if name in classifier_params and param.requires_grad:
                            param.data.copy_(classifier_params[name])
        else:
            # 标准模式：设置所有参数
            with torch.no_grad():
                for name, param in self.classifier.named_parameters():
                    if name in params and param.requires_grad:
                        param.data.copy_(params[name])
    
    def _get_model_device(self):
        """获取模型所在设备 - 带缓存优化"""
        if self._device_cache is None or self._device_cache_dirty:
            if hasattr(self.classifier, 'parameters'):
                try:
                    self._device_cache = next(self.classifier.parameters()).device
                    self._device_cache_dirty = False
                except StopIteration:
                    self._device_cache = torch.device('cpu')
            elif hasattr(self.image_encoder, 'parameters'):
                try:
                    self._device_cache = next(self.image_encoder.parameters()).device
                    self._device_cache_dirty = False
                except StopIteration:
                    self._device_cache = torch.device('cpu')
            else:
                self._device_cache = torch.device('cpu')
        return self._device_cache
    
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
        
        # 获取缓存的设备，避免重复调用
        device = self._get_model_device()
        data, labels = device_manager.move_tensors_to_device(data, labels, device=device)
        
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
    
    def evaluate_with_dataloader(self, data_loader) -> Dict[str, float]:
        """
        使用数据加载器评估模型
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            评估指标字典
        """
        self.classifier.eval()
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        top5_correct = 0
        
        # 获取缓存的设备，避免每次batch都重复调用
        device = self._get_model_device()
        
        with torch.no_grad():
            for batch_data, batch_labels in data_loader:
                # 简化设备移动操作
                batch_data, batch_labels = device_manager.move_tensors_to_device(
                    batch_data, batch_labels, device=device
                )
                
                # 前向传播
                outputs = self.classifier(batch_data)
                loss = self.criterion(outputs, batch_labels)
                
                # 修正：使用样本数加权平均
                total_loss += loss.item() * batch_data.size(0)
                total_samples += batch_data.size(0)
                
                # 计算准确率
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == batch_labels).sum().item()
                
                # 计算Top-5准确率（如果类别数>=5）
                if self.num_classes >= 5:
                    _, top5_pred = outputs.topk(5, 1, largest=True, sorted=True)
                    top5_correct += top5_pred.eq(batch_labels.view(-1, 1).expand_as(top5_pred)).sum().item()
        
        # 计算平均指标，使用样本数加权平均
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        result = {
            'loss': avg_loss,
            'accuracy': accuracy
        }
        
        # 添加Top-5准确率（如果适用）
        if self.num_classes >= 5:
            top5_accuracy = top5_correct / total_samples if total_samples > 0 else 0.0
            result['top5_accuracy'] = top5_accuracy
            
        return result
    
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
        
        summary = {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'encoder_feature_dim': self.image_encoder.feature_dim,
            'normalize_features': self.normalize_features,
            'lora_enabled': self._lora_enabled
        }
        
        # 添加LoRA特定信息
        if self._lora_enabled and self.lora_wrapper is not None:
            lora_trainable_params = self.lora_wrapper.get_trainable_parameters()
            summary.update({
                'lora_trainable_parameters': lora_trainable_params,
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


# 统一的工厂函数，支持从配置或checkpoint创建CLIP模型
def create_clip_model(config: Dict[str, Any]) -> FederatedCLIPModel:
    """
    创建联邦学习CLIP模型的统一工厂函数
    
    Args:
        config: 模型配置字典，可以包含以下键：
            - model_name: 预训练模型名称
            - num_classes: 分类类别数
            - normalize_features: 是否对特征进行归一化
            - freeze_encoder: 是否冻结编码器
            - cache_dir: 模型缓存目录
            - optimizer_config: 优化器配置
            - checkpoint_path: 如果提供，将从此路径加载模型权重
            - lora: LoRA配置字典
        
    Returns:
        联邦学习CLIP模型实例
    """
    # 如果提供了checkpoint路径，优先使用from_checkpoint方法
    if 'checkpoint_path' in config and config['checkpoint_path'] is not None:
        checkpoint_path = config.pop('checkpoint_path')
        return FederatedCLIPModel.from_checkpoint(checkpoint_path, **config)
    
    # 否则直接创建新模型
    return FederatedCLIPModel(
        model_name=config.get('model_name', 'openai/clip-vit-base-patch32'),
        num_classes=config.get('num_classes', 10),
        normalize_features=config.get('normalize_features', True),
        freeze_encoder=config.get('freeze_encoder', False),
        cache_dir=config.get('cache_dir', None),
        optimizer_config=config.get('optimizer_config', None),
        lora_config=config.get('lora', None)
    )
