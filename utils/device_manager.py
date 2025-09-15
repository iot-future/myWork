"""
设备管理工具类
提供统一的GPU/CPU设备管理功能
"""

import torch
import warnings
from typing import Union, Optional, Dict, Any, Tuple


class DeviceManager:
    """设备管理器 - 统一管理GPU/CPU设备相关操作"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._current_device = None
        self._initialized = True
    
    def get_optimal_device(self, preference: str = 'auto') -> torch.device:
        """获取最优设备"""
        if preference == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        elif preference == 'cpu':
            device = torch.device('cpu')
        elif preference == 'cuda':
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                warnings.warn("CUDA not available, using CPU", UserWarning)
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
        
        self._current_device = device
        return device
    
    
    def move_model_to_device(self, model, device: Optional[torch.device] = None):
        """将模型移动到指定设备"""
        target_device = device or self._current_device or self.get_optimal_device()
        
        try:
            # 检查是否是 BaseModel 实例
            if hasattr(model, 'model') and hasattr(model.model, 'to'):
                # 对于 BaseModel，移动内部的 model
                model.model = model.model.to(target_device)
                return model
            elif hasattr(model, 'to'):
                # 对于普通的 nn.Module
                return model.to(target_device)
            else:
                # 如果对象没有 to 方法，尝试移动到 CPU 作为回退
                if hasattr(model, 'model') and hasattr(model.model, 'to'):
                    model.model = model.model.to('cpu')
                return model
        except Exception:
            # 发生异常时的回退处理
            if hasattr(model, 'model') and hasattr(model.model, 'to'):
                model.model = model.model.to('cpu')
            return model
    
    def move_tensors_to_device(self, *tensors: torch.Tensor, device: Optional[torch.device] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """将张量移动到指定设备"""
        target_device = device or self._current_device or self.get_optimal_device()
        
        moved_tensors = []
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                try:
                    moved_tensors.append(tensor.to(target_device))
                except Exception:
                    moved_tensors.append(tensor.to('cpu'))
            else:
                moved_tensors.append(tensor)
        
        return moved_tensors[0] if len(moved_tensors) == 1 else tuple(moved_tensors)
    
    def get_current_device(self) -> Optional[torch.device]:
        """获取当前设备"""
        return self._current_device


# 全局设备管理器实例
device_manager = DeviceManager()


def get_device(preference: str = 'auto') -> torch.device:
    """获取最优设备的便捷函数"""
    return device_manager.get_optimal_device(preference)


def to_device(obj, device: Optional[torch.device] = None):
    """将对象移动到设备的便捷函数"""
    if isinstance(obj, torch.nn.Module) or hasattr(obj, 'model'):
        return device_manager.move_model_to_device(obj, device)
    elif isinstance(obj, torch.Tensor):
        return device_manager.move_tensors_to_device(obj, device=device)
    else:
        return obj
