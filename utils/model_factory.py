"""
模型工厂模块
负责根据配置创建不同类型的模型，支持配置化优化器
"""

from typing import Dict, Any
from models.cnn import CNNModel
from models.base import SimpleLinearModel
from models.clip import CLIPModel


class ModelFactory:
    """模型工厂类"""
    
    @staticmethod
    def create_model(model_config: Dict[str, Any], optimizer_config: Dict[str, Any] = None):
        """
        根据配置创建模型
        
        Args:
            model_config: 模型配置字典
            optimizer_config: 优化器配置字典
            
        Returns:
            配置好的模型实例
        """
        model_type = model_config['type']
        
        if model_type == 'cnn':
            return CNNModel(optimizer_config=optimizer_config)
        elif model_type == 'linear':
            # 线性模型需要额外的维度参数
            input_dim = model_config.get('input_dim', 784)  # MNIST默认28*28
            output_dim = model_config.get('output_dim', 10)  # 10个类别
            return SimpleLinearModel(
                input_dim=input_dim, 
                output_dim=output_dim,
                optimizer_config=optimizer_config
            )
        elif model_type == 'clip':
            # 完整版CLIP模型
            return CLIPModel(
                img_size=model_config.get('img_size', 224),
                patch_size=model_config.get('patch_size', 32),
                in_channels=model_config.get('in_channels', 3),
                vocab_size=model_config.get('vocab_size', 50000),
                max_text_len=model_config.get('max_text_len', 77),
                d_model=model_config.get('d_model', 512),
                n_layers=model_config.get('n_layers', 12),
                n_heads=model_config.get('n_heads', 8),
                d_ff=model_config.get('d_ff', 2048),
                dropout=model_config.get('dropout', 0.1),
                temperature=model_config.get('temperature', 0.07),
                optimizer_config=optimizer_config
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    @staticmethod
    def get_supported_models():
        """获取支持的模型类型列表"""
        return ['cnn', 'linear', 'clip']
