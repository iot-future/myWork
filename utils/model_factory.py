"""
模型工厂模块
负责根据配置创建不同类型的模型，支持配置化优化器
"""

from typing import Dict, Any, Optional, List
from models.cnn import FederatedCNNModel
from models.base import FederatedLinearModel
from models.clip import FederatedCLIPModel


class ModelFactory:
    """模型工厂类"""

    @staticmethod
    def create_model(model_config: Dict[str, Any], optimizer_config: Optional[Dict[str, Any]] = None,
                     dataset_name: Optional[List[str]] = None):
        """
        根据配置创建模型
        
        Args:
            model_config: 模型配置字典
            optimizer_config: 优化器配置字典
            dataset_name: 所用的数据集
            
        Returns:
            配置好的模型实例
        """
        model_type = model_config['type']

        if model_type == 'cnn':
            return FederatedCNNModel(optimizer_config=optimizer_config)
        elif model_type == 'linear':
            # 线性模型需要额外的维度参数
            input_dim = model_config.get('input_dim', 784)  # MNIST默认28*28
            output_dim = model_config.get('output_dim', 10)  # 10个类别
            return FederatedLinearModel(
                input_dim=input_dim,
                output_dim=output_dim,
                optimizer_config=optimizer_config
            )
        elif model_type == 'clip':
            # 检查是否启用LoRA
            lora_config = model_config.get('lora', {})
            if lora_config.get('enabled', False):
                print(f"🔬 CLIP模型启用LoRA微调 (r={lora_config.get('r', 16)})")

            # 基于Hugging Face的CLIP模型
            return FederatedCLIPModel(
                model_name=model_config.get('model_name', 'openai/clip-vit-base-patch32'),
                num_classes=model_config.get('num_classes', 10),
                normalize_features=model_config.get('normalize_features', True),
                freeze_encoder=model_config.get('freeze_vision_encoder', False),
                cache_dir=model_config.get('cache_dir', None),
                optimizer_config=optimizer_config,
                lora_config=lora_config if lora_config.get('enabled', False) else None,
                dataset_name=dataset_name
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    @staticmethod
    def get_supported_models():
        """获取支持的模型类型列表"""
        return ['cnn', 'linear', 'clip']

    @staticmethod
    def get_model_info(model_type: str) -> Dict[str, Any]:
        """
        获取特定模型类型的信息
        
        Args:
            model_type: 模型类型
            
        Returns:
            模型信息字典
        """
        if model_type == 'cnn':
            return {
                'name': 'CNN模型',
                'description': '卷积神经网络，适用于图像分类任务',
                'required_params': [],
                'optional_params': ['optimizer_config']
            }
        elif model_type == 'linear':
            return {
                'name': '线性模型',
                'description': '简单的全连接神经网络',
                'required_params': [],
                'optional_params': ['input_dim', 'output_dim', 'optimizer_config']
            }
        elif model_type == 'clip':
            return {
                'name': 'CLIP模型',
                'description': '基于Hugging Face的CLIP视觉-语言多模态模型',
                'required_params': [],
                'optional_params': [
                    'model_name', 'num_classes', 'normalize_features',
                    'freeze_vision_encoder', 'cache_dir', 'optimizer_config'
                ]
            }
        else:
            return {
                'name': '未知模型',
                'description': '不支持的模型类型',
                'required_params': [],
                'optional_params': []
            }
