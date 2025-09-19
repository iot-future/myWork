"""
CLIP模型专用LoRA包装器

为CLIP模型的图像编码器和文本编码器提供LoRA微调功能
"""

from typing import Dict, Any, Optional, List
import torch
from transformers import CLIPVisionModel, CLIPTextModel

from .loRA_wrapper import LoRAWrapper


class CLIPLoRAWrapper:
    """CLIP模型专用LoRA包装器
    
    为CLIP的图像编码器和文本编码器分别提供LoRA微调功能
    """
    
    def __init__(self, 
                 vision_model: Optional[CLIPVisionModel] = None,
                 text_model: Optional[CLIPTextModel] = None):
        """初始化CLIP LoRA包装器
        
        Args:
            vision_model: CLIP视觉编码器模型
            text_model: CLIP文本编码器模型
        """
        self.vision_model = vision_model
        self.text_model = text_model
        
        # LoRA包装器
        self.vision_lora_wrapper = None
        self.text_lora_wrapper = None
    
    def apply_lora(self,
                   vision_config: Optional[Dict[str, Any]] = None,
                   text_config: Optional[Dict[str, Any]] = None) -> None:
        """为视觉和文本编码器应用LoRA
        
        Args:
            vision_config: 视觉编码器LoRA配置
            text_config: 文本编码器LoRA配置
        """
        # 默认配置
        default_config = {
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'target_modules': ["q_proj", "v_proj", "k_proj", "out_proj"]
        }
        
        # 应用视觉编码器LoRA
        if self.vision_model is not None:
            config = {**default_config, **(vision_config or {})}
            self.vision_lora_wrapper = LoRAWrapper(self.vision_model)
            self.vision_lora_wrapper.apply_lora(**config)
        
        # 应用文本编码器LoRA
        if self.text_model is not None:
            config = {**default_config, **(text_config or {})}
            self.text_lora_wrapper = LoRAWrapper(self.text_model)
            self.text_lora_wrapper.apply_lora(**config)
    
    def get_lora_parameters(self) -> Dict[str, torch.Tensor]:
        """获取所有LoRA参数"""
        lora_params = {}
        
        # 获取视觉编码器LoRA参数
        if self.vision_lora_wrapper and self.vision_lora_wrapper.is_lora_applied():
            vision_params = self.vision_lora_wrapper.get_lora_parameters()
            for name, param in vision_params.items():
                lora_params[f"vision.{name}"] = param
        
        # 获取文本编码器LoRA参数
        if self.text_lora_wrapper and self.text_lora_wrapper.is_lora_applied():
            text_params = self.text_lora_wrapper.get_lora_parameters()
            for name, param in text_params.items():
                lora_params[f"text.{name}"] = param
        
        return lora_params
    
    def set_lora_parameters(self, lora_params: Dict[str, torch.Tensor]) -> None:
        """设置LoRA参数"""
        # 分离视觉和文本编码器参数
        vision_params = {}
        text_params = {}
        
        for name, param in lora_params.items():
            if name.startswith("vision."):
                vision_params[name[7:]] = param  # 移除"vision."前缀
            elif name.startswith("text."):
                text_params[name[5:]] = param   # 移除"text."前缀
        
        # 设置参数
        if vision_params and self.vision_lora_wrapper:
            self.vision_lora_wrapper.set_lora_parameters(vision_params)
        
        if text_params and self.text_lora_wrapper:
            self.text_lora_wrapper.set_lora_parameters(text_params)
    
    def get_trainable_parameters(self) -> int:
        """获取总的可训练参数数量"""
        total = 0
        if self.vision_lora_wrapper and self.vision_lora_wrapper.is_lora_applied():
            total += self.vision_lora_wrapper.get_trainable_parameters()
        if self.text_lora_wrapper and self.text_lora_wrapper.is_lora_applied():
            total += self.text_lora_wrapper.get_trainable_parameters()
        return total
    
    def is_lora_applied(self) -> Dict[str, bool]:
        """检查LoRA应用状态"""
        return {
            'vision': self.vision_lora_wrapper is not None and self.vision_lora_wrapper.is_lora_applied(),
            'text': self.text_lora_wrapper is not None and self.text_lora_wrapper.is_lora_applied()
        }
    
    def get_trainable_parameters(self) -> int:
        """获取可训练参数数量"""
        total_params = 0
        
        if self.vision_lora_wrapper and self.vision_lora_wrapper.is_lora_applied():
            for param in self.vision_lora_wrapper.lora_model.parameters():
                if param.requires_grad:
                    total_params += param.numel()
        
        if self.text_lora_wrapper and self.text_lora_wrapper.is_lora_applied():
            for param in self.text_lora_wrapper.lora_model.parameters():
                if param.requires_grad:
                    total_params += param.numel()
        
        return total_params