"""
LoRA 基础包装器模块
"""

from typing import Dict, Any, Optional
import torch
from peft import LoraConfig, get_peft_model
from copy import deepcopy
TORCH_AVAILABLE = True


class LoRAWrapper:
    """LoRA包装器 - 基于PEFT库
    
    使用Hugging Face PEFT库实现LoRA功能，采用装饰器模式包装现有模型。
    """

    def __init__(self, model: Any):
        """初始化LoRA包装器
        
        Args:
            model: 原始PyTorch模型
        """
        if not TORCH_AVAILABLE:
            raise ImportError("torch and peft are required for LoRAWrapper")

        self.model = model
        self._is_lora_applied = False

    def apply_lora(self,
                   r: int = 16,
                   lora_alpha: int = 32,
                   lora_dropout: float = 0.1,
                   target_modules: Optional[list] = None) -> None:
        """应用LoRA到模型
        
        Args:
            r: LoRA rank
            lora_alpha: LoRA scaling参数
            lora_dropout: dropout概率
            target_modules: 目标模块列表
        """
        if self._is_lora_applied:
            print("LoRA is already applied.")
            return

        try:
            # 设置默认目标模块
            if target_modules is None:
                target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]

            # 创建PEFT LoraConfig
            peft_config = LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="FEATURE_EXTRACTION"
            )
            # 应用LoRA
            self.model = get_peft_model(self.model, peft_config)
            self._is_lora_applied = True

            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())

            print(f"✅ LoRA应用成功 | 可训练参数: {trainable_params:,} ({(trainable_params / total_params) * 100:.2f}%)")

        except Exception as e:
            print(f"Failed to apply LoRA: {e}")
            raise

    def get_lora_parameters(self) -> Dict[str, Any]:
        """获取LoRA参数
        
        Returns:
            LoRA参数字典，只包含LoRA相关的参数，结构为 {参数名: 参数值}
        """
        if not self._is_lora_applied or self.model is None:
            return {}

        lora_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'lora_' in name:
                lora_params[name] = param.data.clone()

        return lora_params

    def get_trainable_parameters(self) -> int:
        """获取可训练参数数量"""
        if not self._is_lora_applied or self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def set_lora_parameters(self, lora_params: Dict[str, Any]) -> None:
        """设置LoRA参数
        
        Args:
            lora_params: LoRA参数字典
        """
        if not self._is_lora_applied or self.model is None:
            print("LoRA is not applied. Cannot set LoRA parameters.")
            return
        
        # 深拷贝LoRA参数，以避免修改原始参数，但是copy_本身是in-place操作，用于以防万一
        lora_params = deepcopy(lora_params)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in lora_params and param.requires_grad:
                    param.data.copy_(lora_params[name])

    def is_lora_applied(self) -> bool:
        """检查LoRA是否已应用"""
        return self._is_lora_applied

    def __getattr__(self, name: str):
        """委托属性访问到原始模型或LoRA模型"""
        if self._is_lora_applied and self.model is not None:
            return getattr(self.model, name)
        else:
            return getattr(self.model, name)
