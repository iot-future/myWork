"""
优化器工厂模块
精简且可复用的优化器配置和创建系统
"""

import torch.optim as optim
from typing import Dict, Any, Type


class OptimizerFactory:
    """优化器工厂类 - 精简且可复用"""
    
    # 支持的优化器映射
    OPTIMIZERS = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'rmsprop': optim.RMSprop,
    }
    
    @classmethod
    def create_optimizer(cls, model_parameters, config: Dict[str, Any]):
        """
        根据配置创建优化器
        
        Args:
            model_parameters: 模型参数
            config: 优化器配置字典
            
        Returns:
            配置好的优化器实例
        """
        optimizer_type = config['type'].lower()
        
        if optimizer_type not in cls.OPTIMIZERS:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
        
        # 基本参数
        optimizer_class = cls.OPTIMIZERS[optimizer_type]
        learning_rate = config['learning_rate']
        
        # 根据优化器类型设置特定参数
        kwargs = {'lr': learning_rate}
        
        if optimizer_type == 'sgd':
            kwargs.update(cls._get_sgd_params(config))
        elif optimizer_type in ['adam', 'adamw']:
            kwargs.update(cls._get_adam_params(config))
        elif optimizer_type == 'rmsprop':
            kwargs.update(cls._get_rmsprop_params(config))
        
        return optimizer_class(model_parameters, **kwargs)
    
    @staticmethod
    def _get_sgd_params(config: Dict[str, Any]) -> Dict[str, Any]:
        """获取SGD优化器参数"""
        params = {}
        if 'momentum' in config:
            params['momentum'] = config['momentum']
        if 'weight_decay' in config:
            params['weight_decay'] = config['weight_decay']
        return params
    
    @staticmethod
    def _get_adam_params(config: Dict[str, Any]) -> Dict[str, Any]:
        """获取Adam/AdamW优化器参数"""
        params = {}
        if 'betas' in config:
            params['betas'] = tuple(config['betas'])
        if 'eps' in config:
            params['eps'] = config['eps']
        if 'weight_decay' in config:
            params['weight_decay'] = config['weight_decay']
        return params
    
    @staticmethod
    def _get_rmsprop_params(config: Dict[str, Any]) -> Dict[str, Any]:
        """获取RMSprop优化器参数"""
        params = {}
        if 'alpha' in config:
            params['alpha'] = config['alpha']
        if 'eps' in config:
            params['eps'] = config['eps']
        if 'weight_decay' in config:
            params['weight_decay'] = config['weight_decay']
        if 'momentum' in config:
            params['momentum'] = config['momentum']
        return params
    
    @classmethod
    def get_supported_optimizers(cls):
        """获取支持的优化器类型列表"""
        return list(cls.OPTIMIZERS.keys())
    
    @classmethod
    def get_default_config(cls, optimizer_type: str) -> Dict[str, Any]:
        """获取指定优化器的默认配置"""
        defaults = {
            'sgd': {
                'type': 'sgd',
                'learning_rate': 0.01,
                'momentum': 0.9,
                'weight_decay': 0.0001
            },
            'adam': {
                'type': 'adam',
                'learning_rate': 0.001,
                'betas': [0.9, 0.999],
                'eps': 1e-8,
                'weight_decay': 0.0001
            },
            'adamw': {
                'type': 'adamw',
                'learning_rate': 0.001,
                'betas': [0.9, 0.999],
                'eps': 1e-8,
                'weight_decay': 0.01
            },
            'rmsprop': {
                'type': 'rmsprop',
                'learning_rate': 0.01,
                'alpha': 0.99,
                'eps': 1e-8,
                'weight_decay': 0.0001
            }
        }
        
        return defaults.get(optimizer_type.lower(), {})
