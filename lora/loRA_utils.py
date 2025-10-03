"""
LoRA 基础包装器模块
"""

from typing import Dict, Any, Optional
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from copy import deepcopy

TORCH_AVAILABLE = True


def apply_lora(model=None, r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.1,
               target_modules: Optional[list] = None) -> PeftModel:
    """应用LoRA到模型

    Args:
        model: 原始模型
        r: LoRA rank
        lora_alpha: LoRA scaling参数
        lora_dropout: dropout概率
        target_modules: 目标模块列表

    returns:
        应用LoRA后的模型
    """
    if model is None:
        raise ValueError("Model is None. Cannot apply LoRA.")
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
        lora_model = get_peft_model(model, peft_config)

        trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in lora_model.parameters())

        print(f"✅ LoRA应用成功 | 可训练参数: {trainable_params:,} ({(trainable_params / total_params) * 100:.2f}%)")
        return lora_model

    except Exception as e:
        raise ValueError(f"Failed to apply LoRA: {e}")


def get_lora_parameters(model) -> Dict[str, Any]:
    """获取LoRA参数
    Args:
        model: 包含LoRA的模型实例

    Returns:
        LoRA参数字典，只包含LoRA相关的参数，结构为 {参数名: 参数值}
    """
    if model.lora_model is None:
        return {}

    lora_params = {}
    for name, param in model.lora_model.named_parameters():
        if param.requires_grad and 'lora_' in name:
            lora_params[name] = param.data.clone()

    return lora_params


def get_trainable_parameters(model) -> int:
    """获取可训练参数数量
    Args:
        model: 包含LoRA的模型实例
    Returns:
        可训练参数数量
    """
    if model.lora_model is None:
        return 0
    return sum(p.numel() for p in model.lora_model.parameters() if p.requires_grad)


def set_lora_parameters(model, lora_params: Dict[str, Any]) -> None:
    """设置LoRA参数

    Args:
        model: 包含LoRA的模型实例
        lora_params: LoRA参数字典
    """
    if model.lora_model is None:
        print("LoRA is not applied. Cannot set LoRA parameters.")
        return

    # 深拷贝LoRA参数，以避免修改原始参数，但是copy_本身是in-place操作，用于以防万一
    lora_params = deepcopy(lora_params)
    with torch.no_grad():
        for name, param in model.lora_model.named_parameters():
            if name in lora_params and param.requires_grad:
                param.data.copy_(lora_params[name])


def is_lora_applied(model) -> bool:
    """检查LoRA是否已应用"""
    return model.is_lora_applied
