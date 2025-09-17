"""
LoRA (Low-Rank Adaptation) 配置管理模块

此模块提供 LoRA 参数配置、目标模块配置等功能。
设计目标：与现有联邦学习框架解耦，便于后续集成。
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import torch


@dataclass
class LoRAConfig:
    """LoRA配置类
    
    提供LoRA模型的所有配置参数，支持不同模型类型的配置。
    """
    # LoRA基本参数
    r: int = 16  # LoRA rank，控制低秩分解的维度
    lora_alpha: int = 32  # LoRA scaling 参数
    lora_dropout: float = 0.1  # LoRA层的dropout概率
    
    # 目标模块配置
    target_modules: Optional[Union[List[str], str]] = None  # 要应用LoRA的模块名称
    bias: str = "none"  # bias处理方式: "none", "all", "lora_only"
    
    # 任务类型配置
    task_type: str = "FEATURE_EXTRACTION"  # 任务类型，PEFT库要求
    
    # 推理配置
    inference_mode: bool = False  # 是否为推理模式
    
    # 模型特定配置
    modules_to_save: Optional[List[str]] = None  # 需要保存的完整模块（非LoRA）
    
    # 联邦学习特定配置
    enable_federated: bool = True  # 是否启用联邦学习特性
    save_only_trainable: bool = True  # 是否只保存可训练参数
    
    def __post_init__(self):
        """初始化后处理"""
        if self.target_modules is None:
            # 为CLIP模型设置默认目标模块
            self.target_modules = [
                "q_proj", "v_proj", "k_proj", "out_proj",  # 注意力层
                "fc1", "fc2",  # MLP层
            ]
            
    def to_peft_config(self) -> Dict[str, Any]:
        """转换为PEFT库所需的配置格式
        
        Returns:
            PEFT LoraConfig 所需的配置字典
        """
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type,
            "inference_mode": self.inference_mode,
            "modules_to_save": self.modules_to_save,
        }
    
    @classmethod
    def for_clip_model(cls, r: int = 16, lora_alpha: int = 32) -> "LoRAConfig":
        """为CLIP模型创建专用配置
        
        Args:
            r: LoRA rank
            lora_alpha: LoRA alpha参数
            
        Returns:
            适用于CLIP模型的LoRA配置
        """
        return cls(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=[
                # Vision Transformer层
                "q_proj", "v_proj", "k_proj", "out_proj",
                "fc1", "fc2",
                # 可选：包含层归一化
                # "layernorm_before", "layernorm_after"
            ],
            task_type="FEATURE_EXTRACTION",
            bias="none",
            lora_dropout=0.1,
        )
    
    @classmethod
    def for_classification_head(cls, r: int = 8, lora_alpha: int = 16) -> "LoRAConfig":
        """为分类头创建专用配置
        
        Args:
            r: LoRA rank（分类头通常使用较小的rank）
            lora_alpha: LoRA alpha参数
            
        Returns:
            适用于分类头的LoRA配置
        """
        return cls(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=["classifier", "fc", "linear"],  # 常见的分类器层名称
            task_type="FEATURE_EXTRACTION",
            bias="none",
            lora_dropout=0.05,  # 分类头使用较小的dropout
        )


@dataclass
class LoRAModelState:
    """LoRA模型状态类
    
    管理LoRA模型的状态信息，包括训练状态、权重状态等。
    """
    is_lora_enabled: bool = False  # LoRA是否已启用
    original_parameters_count: int = 0  # 原始模型参数数量
    lora_parameters_count: int = 0  # LoRA参数数量
    trainable_parameters_count: int = 0  # 可训练参数数量
    
    # 模型状态
    is_frozen: bool = False  # 原始权重是否被冻结
    lora_modules: List[str] = field(default_factory=list)  # 应用了LoRA的模块列表
    
    def get_parameter_efficiency(self) -> float:
        """计算参数效率（LoRA参数占原始参数的比例）
        
        Returns:
            参数效率比例 (0-1)
        """
        if self.original_parameters_count == 0:
            return 0.0
        return self.lora_parameters_count / self.original_parameters_count
    
    def get_trainable_ratio(self) -> float:
        """计算可训练参数比例
        
        Returns:
            可训练参数比例 (0-1)
        """
        if self.original_parameters_count == 0:
            return 0.0
        return self.trainable_parameters_count / self.original_parameters_count


class LoRAUtils:
    """LoRA工具类
    
    提供LoRA相关的实用函数，如参数统计、设备管理等。
    """
    
    @staticmethod
    def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
        """统计模型参数数量
        
        Args:
            model: PyTorch模型
            trainable_only: 是否只统计可训练参数
            
        Returns:
            参数数量
        """
        if trainable_only:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in model.parameters())
    
    @staticmethod
    def get_lora_module_names(model: torch.nn.Module) -> List[str]:
        """获取模型中的LoRA模块名称
        
        Args:
            model: 应用了LoRA的模型
            
        Returns:
            LoRA模块名称列表
        """
        lora_modules = []
        for name, module in model.named_modules():
            # 检查是否为LoRA相关模块
            if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                lora_modules.append(name)
        return lora_modules
    
    @staticmethod
    def print_model_info(model: torch.nn.Module, title: str = "Model Info") -> None:
        """打印模型信息
        
        Args:
            model: PyTorch模型
            title: 信息标题
        """
        total_params = LoRAUtils.count_parameters(model, trainable_only=False)
        trainable_params = LoRAUtils.count_parameters(model, trainable_only=True)
        lora_modules = LoRAUtils.get_lora_module_names(model)
        
        print(f"\n=== {title} ===")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
        if lora_modules:
            print(f"LoRA modules: {len(lora_modules)}")
            for module_name in lora_modules[:5]:  # 只显示前5个
                print(f"  - {module_name}")
            if len(lora_modules) > 5:
                print(f"  ... and {len(lora_modules) - 5} more")
        print("=" * (len(title) + 8))