"""
优化器工厂模块
统一使用AdamW优化器
"""

import torch.optim as optim
from typing import Dict, Any


class OptimizerFactory:
    """优化器工厂类 - 统一使用AdamW"""
    
    @classmethod
    def create_optimizer(cls, model_parameters, config: Dict[str, Any]):
        """
        创建AdamW优化器
        
        Args:
            model_parameters: 模型参数
            config: 优化器配置字典
            
        Returns:
            配置好的AdamW优化器实例
        """
        # 默认AdamW参数，确保类型转换
        learning_rate = float(config.get('learning_rate', 0.001))
        weight_decay = float(config.get('weight_decay', 0.01))
        betas = config.get('betas', [0.9, 0.999])
        # 确保betas中的值都是浮点数
        if isinstance(betas, list):
            betas = [float(b) for b in betas]
        eps = float(config.get('eps', 0.00000001))
        
        return optim.AdamW(
            model_parameters,
            lr=learning_rate,
            betas=tuple(betas),
            eps=eps,
            weight_decay=weight_decay
        )
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """获取AdamW的默认配置"""
        return {
            'type': 'adamw',
            'learning_rate': 0.001,
            'betas': [0.9, 0.999],
            'eps': 1e-8,
            'weight_decay': 0.01
        }