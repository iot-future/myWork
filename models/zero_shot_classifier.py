"""
零样本分类头实现
基于CLIP文本编码器和预定义模板构建分类头，与CLIP模型解耦
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union
from data.templates import get_templates
from utils.device_manager import device_manager


class  ZeroShotClassificationHead(nn.Module):
    """零样本分类头基类"""
    
    def __init__(self, text_encoder, dataset_name: str, class_names: List[str], 
                 input_dim: int = 768, temperature: float = 1.0):
        """
        初始化零样本分类头
        
        Args:
            text_encoder: CLIP文本编码器实例
            dataset_name: 数据集名称，用于获取对应的文本模板
            class_names: 类别名称列表
            input_dim: 输入特征维度（图像编码器输出维度，默认768）
            temperature: 温度参数，用于logits缩放
        """
        super().__init__()
        
        self.text_encoder = text_encoder
        self.dataset_name = dataset_name
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.input_dim = input_dim
        self.temperature = temperature
        
        # 获取数据集对应的文本模板
        try:
            self.templates = get_templates(dataset_name)
        except AssertionError:
            # 对于不支持的数据集，使用通用的ImageNet模板
            print(f"数据集 '{dataset_name}' 未定义专用模板")
        
        # 构建零样本权重矩阵
        self.register_buffer('weight_matrix', self._build_weight_matrix())
        
        # 特征维度映射层（768 -> 512）
        self.feature_projection = nn.Linear(input_dim, text_encoder.feature_dim, bias=False)
        nn.init.xavier_uniform_(self.feature_projection.weight)
    
    def _build_weight_matrix(self) -> torch.Tensor:
        """构建零样本分类头的权重矩阵"""
        print(f"构建 {self.dataset_name} 数据集的零样本分类头权重...")
        
        all_class_features = []
        
        for class_name in self.class_names:
            # 为每个类别生成所有模板文本
            class_texts = [template(class_name) for template in self.templates]
            
            # 编码文本为特征
            with torch.no_grad():
                text_features = self.text_encoder(class_texts)
                
                # 对多个模板特征求平均并归一化
                avg_features = text_features.mean(dim=0)
                normalized_features = F.normalize(avg_features, dim=-1, p=2)
                
                all_class_features.append(normalized_features)
        
        # 组合所有类别特征成权重矩阵 [num_classes, feature_dim]
        weight_matrix = torch.stack(all_class_features, dim=0)
        
        print(f"权重矩阵构建完成: {weight_matrix.shape}")
        return weight_matrix
    
    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            image_features: 图像特征 [batch_size, 768]
            
        Returns:
            logits: 分类logits [batch_size, num_classes]
        """
        # 投影图像特征到文本特征空间 [batch_size, 768] -> [batch_size, 512]
        projected_features = self.feature_projection(image_features)
        
        # 归一化投影后的特征
        normalized_features = F.normalize(projected_features, dim=-1, p=2)
        
        # 计算相似度得分
        logits = torch.matmul(normalized_features, self.weight_matrix.t()) / self.temperature
        
        return logits
    
    def update_class_names(self, new_class_names: List[str]):
        """更新类别名称并重新构建权重矩阵"""
        self.class_names = new_class_names
        self.num_classes = len(new_class_names)
        self.weight_matrix = self._build_weight_matrix()
    
    def get_class_similarities(self, image_features: torch.Tensor) -> torch.Tensor:
        """获取图像特征与各类别的相似度"""
        with torch.no_grad():
            projected_features = self.feature_projection(image_features)
            normalized_features = F.normalize(projected_features, dim=-1, p=2)
            similarities = torch.matmul(normalized_features, self.weight_matrix.t())
            return similarities


class CIFAR10ZeroShotHead(ZeroShotClassificationHead):
    """CIFAR-10专用零样本分类头"""
    
    def __init__(self, text_encoder, temperature: float = 1.0):
        cifar10_classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        super().__init__(text_encoder, 'CIFAR10', cifar10_classes, temperature=temperature)


class CIFAR100ZeroShotHead(ZeroShotClassificationHead):
    """CIFAR-100专用零样本分类头"""
    
    def __init__(self, text_encoder, temperature: float = 1.0):
        # CIFAR-100的100个类别
        cifar100_classes = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
            'worm'
        ]
        super().__init__(text_encoder, 'CIFAR100', cifar100_classes, temperature=temperature)


class MNISTZeroShotHead(ZeroShotClassificationHead):
    """MNIST专用零样本分类头"""
    
    def __init__(self, text_encoder, temperature: float = 1.0):
        mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        super().__init__(text_encoder, 'MNIST', mnist_classes, temperature=temperature)


def create_zero_shot_head(dataset_name: str, text_encoder, class_names: Optional[List[str]] = None, 
                         temperature: float = 1.0) -> ZeroShotClassificationHead:
    """
    零样本分类头工厂函数
    
    Args:
        dataset_name: 数据集名称
        text_encoder: 文本编码器实例
        class_names: 类别名称列表（某些数据集需要）
        temperature: 温度参数
        
    Returns:
        对应的零样本分类头实例
    """
    dataset_name_upper = dataset_name.upper()
    
    if dataset_name_upper == 'CIFAR10':
        return CIFAR10ZeroShotHead(text_encoder, temperature)
    elif dataset_name_upper == 'CIFAR100':
        return CIFAR100ZeroShotHead(text_encoder, temperature)
    elif dataset_name_upper == 'MNIST':
        return MNISTZeroShotHead(text_encoder, temperature)
    else:
        # 通用零样本分类头
        if class_names is None:
            raise ValueError(f"未知数据集 {dataset_name} 需要提供class_names参数")
        return ZeroShotClassificationHead(text_encoder, dataset_name, class_names, temperature)


class MultiDatasetZeroShotClassifier(nn.Module):
    """
    多数据集零样本分类器
    支持一个客户端同时处理多个数据集，为每个数据集维护独立的分类头
    """
    
    def __init__(self, image_encoder, text_encoder, temperature: float = 1.0):
        """
        初始化多数据集零样本分类器
        
        Args:
            image_encoder: 图像编码器
            text_encoder: 文本编码器
            temperature: 温度参数
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.temperature = temperature
        
        # 存储各数据集的零样本分类头
        self.classification_heads = nn.ModuleDict()
        
        # 记录已注册的数据集
        self.registered_datasets = set()
        
        # 冻结权重矩阵，但保持特征投影层可训练
        self._freeze_weight_matrices()
    
    def register_dataset(self, dataset_name: str, class_names: Optional[List[str]] = None):
        """
        注册新数据集并创建对应的零样本分类头
        
        Args:
            dataset_name: 数据集名称
            class_names: 类别名称列表（某些数据集需要）
        """
        if dataset_name in self.registered_datasets:
            print(f"数据集 {dataset_name} 已经注册，跳过重复注册")
            return
        
        print(f"注册数据集: {dataset_name}")
        
        # 创建零样本分类头
        zero_shot_head = create_zero_shot_head(
            dataset_name=dataset_name,
            text_encoder=self.text_encoder,
            class_names=class_names,
            temperature=self.temperature
        )
        
        # 添加到模块字典
        self.classification_heads[dataset_name] = zero_shot_head
        self.registered_datasets.add(dataset_name)
        
        # 冻结新添加分类头的权重矩阵，保持特征投影层可训练
        self._freeze_weight_matrix(dataset_name)
        
        print(f"数据集 {dataset_name} 注册完成，类别数: {zero_shot_head.num_classes}")
    
    def _freeze_weight_matrices(self):
        """冻结权重矩阵，但保持特征投影层可训练"""
        for dataset_name in self.classification_heads:
            self._freeze_weight_matrix(dataset_name)
    
    def _freeze_weight_matrix(self, dataset_name: str):
        """冻结指定数据集的权重矩阵，但保持特征投影层可训练"""
        if dataset_name in self.classification_heads:
            head = self.classification_heads[dataset_name]
            # 权重矩阵已经是buffer，不需要梯度
            # 只需要确保特征投影层是可训练的
            for param in head.feature_projection.parameters():
                param.requires_grad = True
            print(f"数据集 {dataset_name}: 权重矩阵冻结，特征投影层可训练")
    
    def _freeze_classification_head(self, dataset_name: str):
        """完全冻结指定数据集分类头的参数（向后兼容）"""
        if dataset_name in self.classification_heads:
            for param in self.classification_heads[dataset_name].parameters():
                param.requires_grad = False
            print(f"已完全冻结数据集 {dataset_name} 的分类头参数")
    
    def unfreeze_classification_head(self, dataset_name: str):
        """解冻指定数据集分类头的参数（如果需要微调）"""
        if dataset_name in self.classification_heads:
            for param in self.classification_heads[dataset_name].parameters():
                param.requires_grad = True
            print(f"已解冻数据集 {dataset_name} 的分类头参数")
    
    def forward(self, images: torch.Tensor, dataset_name: str) -> torch.Tensor:
        """
        前向传播
        
        Args:
            images: 输入图像
            dataset_name: 数据集名称
            
        Returns:
            logits: 分类logits
        """
        if dataset_name not in self.registered_datasets:
            raise ValueError(f"数据集 {dataset_name} 未注册，请先调用 register_dataset()")
        
        # 获取图像特征
        image_features = self.image_encoder(images)
        
        # 使用对应数据集的分类头
        logits = self.classification_heads[dataset_name](image_features)
        
        return logits
    
    def forward_batch(self, images: torch.Tensor, dataset_names: List[str]) -> Dict[str, torch.Tensor]:
        """
        批量前向传播，支持混合数据集
        
        Args:
            images: 输入图像批次
            dataset_names: 每个样本对应的数据集名称列表
            
        Returns:
            Dict[dataset_name, logits]: 按数据集分组的logits
        """
        if len(dataset_names) != images.size(0):
            raise ValueError("dataset_names长度必须与batch_size相等")
        
        # 获取图像特征
        image_features = self.image_encoder(images)
        
        # 按数据集分组处理
        results = {}
        dataset_indices = {}
        
        # 收集每个数据集的样本索引
        for i, dataset_name in enumerate(dataset_names):
            if dataset_name not in dataset_indices:
                dataset_indices[dataset_name] = []
            dataset_indices[dataset_name].append(i)
        
        # 为每个数据集计算logits
        for dataset_name, indices in dataset_indices.items():
            if dataset_name not in self.registered_datasets:
                raise ValueError(f"数据集 {dataset_name} 未注册")
            
            # 提取对应样本的特征
            dataset_features = image_features[indices]
            
            # 计算logits
            dataset_logits = self.classification_heads[dataset_name](dataset_features)
            results[dataset_name] = dataset_logits
        
        return results
    
    def predict(self, images: torch.Tensor, dataset_name: str) -> torch.Tensor:
        """预测类别"""
        with torch.no_grad():
            logits = self.forward(images, dataset_name)
            return torch.argmax(logits, dim=-1)
    
    def predict_proba(self, images: torch.Tensor, dataset_name: str) -> torch.Tensor:
        """预测概率"""
        with torch.no_grad():
            logits = self.forward(images, dataset_name)
            return F.softmax(logits, dim=-1)
    
    def get_similarities(self, images: torch.Tensor, dataset_name: str) -> torch.Tensor:
        """获取相似度得分"""
        if dataset_name not in self.registered_datasets:
            raise ValueError(f"数据集 {dataset_name} 未注册")
        
        with torch.no_grad():
            image_features = self.image_encoder(images)
            return self.classification_heads[dataset_name].get_class_similarities(image_features)
    
    def get_dataset_info(self) -> Dict[str, Dict]:
        """获取所有注册数据集的信息"""
        info = {}
        for dataset_name in self.registered_datasets:
            head = self.classification_heads[dataset_name]
            info[dataset_name] = {
                'num_classes': head.num_classes,
                'class_names': head.class_names,
                'temperature': head.temperature,
                'frozen': not any(p.requires_grad for p in head.parameters())
            }
        return info
    
    def remove_dataset(self, dataset_name: str):
        """移除数据集及其分类头"""
        if dataset_name in self.registered_datasets:
            del self.classification_heads[dataset_name]
            self.registered_datasets.remove(dataset_name)
            print(f"已移除数据集: {dataset_name}")
        else:
            print(f"数据集 {dataset_name} 未注册，无需移除")


class ZeroShotClassifier(nn.Module):
    """单数据集零样本分类器（向后兼容）"""
    
    def __init__(self, image_encoder, zero_shot_head: ZeroShotClassificationHead):
        super().__init__()
        self.image_encoder = image_encoder
        self.zero_shot_head = zero_shot_head
        
        # 冻结分类头参数
        for param in self.zero_shot_head.parameters():
            param.requires_grad = False
    
    def forward(self, images):
        """前向传播"""
        # 获取图像特征
        image_features = self.image_encoder(images)
        # 零样本分类
        logits = self.zero_shot_head(image_features)
        return logits
    
    def predict(self, images):
        """预测类别"""
        with torch.no_grad():
            logits = self.forward(images)
            return torch.argmax(logits, dim=-1)
    
    def predict_proba(self, images):
        """预测概率"""
        with torch.no_grad():
            logits = self.forward(images)
            return F.softmax(logits, dim=-1)
    
    def get_similarities(self, images):
        """获取相似度得分"""
        with torch.no_grad():
            image_features = self.image_encoder(images)
            return self.zero_shot_head.get_class_similarities(image_features)


def create_multi_dataset_classifier(image_encoder, text_encoder, 
                                  dataset_configs: Optional[Dict[str, List[str]]] = None,
                                  temperature: float = 1.0) -> MultiDatasetZeroShotClassifier:
    """
    创建多数据集零样本分类器的便捷函数
    
    Args:
        image_encoder: 图像编码器
        text_encoder: 文本编码器
        dataset_configs: 数据集配置字典 {dataset_name: class_names}
        temperature: 温度参数
        
    Returns:
        MultiDatasetZeroShotClassifier实例
    """
    classifier = MultiDatasetZeroShotClassifier(image_encoder, text_encoder, temperature)
    
    # 如果提供了数据集配置，批量注册
    if dataset_configs:
        for dataset_name, class_names in dataset_configs.items():
            classifier.register_dataset(dataset_name, class_names)
    
    return classifier