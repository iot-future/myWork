"""
LoRA (Low-Rank Adaptation) 模块

提供 LoRA 配置管理、模型包装和实用工具等功能。
"""

from .lora_config import LoRAConfig, LoRAModelState, LoRAUtils

# 导出主要类和函数
__all__ = [
    'LoRAConfig',
    'LoRAModelState', 
    'LoRAUtils',
]

# 版本信息
__version__ = '1.0.0'