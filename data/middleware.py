"""
数据中间件模块 - 加工器版本

专为联邦学习环境设计的数据格式统一中间件。
将不同数据集的DataLoader加工为统一格式的DataLoader：
- 图像：[3, 224, 224] RGB 张量，经过缩放、中心裁剪和ImageNet标准化
- 标签：标量长整型张量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from typing import Any, Callable, Optional
from utils.device_manager import device_manager


class UnifiedDataset(Dataset):
    """
    统一格式的数据集包装器
    
    对原始数据集进行实时加工，输出标准格式：
    - 图像：[3, 224, 224] RGB 张量，ImageNet标准化
    - 标签：标量长整型张量
    """

    def __init__(self,
                 original_dataset: Dataset,
                 dataset_name: str,
                 target_size: int = 224):
        """
        Args:
            original_dataset: 原始数据集
            dataset_name: 数据集名称
            target_size: 目标图像尺寸
        """
        self.original_dataset = original_dataset
        self.dataset_name = dataset_name
        self.target_size = target_size

        # ImageNet预训练模型的标准化参数
        self.imagenet_mean = torch.tensor([0.485, 0.458, 0.408])
        self.imagenet_std = torch.tensor([0.269, 0.261, 0.276])

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        """获取并加工单个样本"""
        image, label = self.original_dataset[idx]

        # 转换为张量（如果还不是的话）
        if not isinstance(image, torch.Tensor):
            import torchvision.transforms as transforms
            image = transforms.ToTensor()(image)

        # 加工图像
        image = self._process_image(image)

        # 加工标签
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = label.long()

        return image, label

    def _process_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        按照指定逻辑加工图像：
        1. 处理通道数（灰度->RGB, RGBA->RGB）
        2. 按最短边缩放到合适尺寸
        3. 中心裁剪到224x224
        4. ImageNet标准化
        """
        # 确保图像为浮点数，范围[0,1]
        if image.dtype == torch.uint8:
            image = image.float() / 255.0

        # 处理维度：如果是HW格式，添加通道维度
        if len(image.shape) == 2:
            image = image.unsqueeze(0)  # H,W -> 1,H,W

        # 处理通道数
        image = self._convert_channels(image)

        # 缩放和裁剪
        image = self._resize_and_crop(image)

        # ImageNet标准化
        image = self._normalize_imagenet(image)

        return image

    def _convert_channels(self, image: torch.Tensor) -> torch.Tensor:
        """处理通道数转换"""
        channels = image.shape[0]

        if channels == 1:
            # 灰度图 -> RGB：复制为3通道
            image = image.repeat(3, 1, 1)
        elif channels == 4:
            # RGBA -> RGB：丢弃Alpha通道
            image = image[:3]
        elif channels == 3:
            # RGB图：保持不变
            pass
        else:
            raise ValueError(f"不支持的通道数: {channels}")

        return image

    def _resize_and_crop(self, image: torch.Tensor) -> torch.Tensor:
        """
        按最短边缩放并中心裁剪到224x224
        """
        c, h, w = image.shape

        # 找到最短边并计算缩放比例
        min_side = min(h, w)
        scale_ratio = self.target_size / min_side

        # 计算缩放后的尺寸
        new_h = int(h * scale_ratio)
        new_w = int(w * scale_ratio)

        # 使用双三次插值进行缩放
        image = image.unsqueeze(0)  # 添加batch维度: C,H,W -> 1,C,H,W
        image = F.interpolate(
            image,
            size=(new_h, new_w),
            mode='bicubic',
            align_corners=False
        )
        image = image.squeeze(0)  # 移除batch维度: 1,C,H,W -> C,H,W

        # 中心裁剪到224x224
        crop_h = crop_w = self.target_size
        start_h = (new_h - crop_h) // 2
        start_w = (new_w - crop_w) // 2

        image = image[:, start_h:start_h + crop_h, start_w:start_w + crop_w]

        return image

    def _normalize_imagenet(self, image: torch.Tensor) -> torch.Tensor:
        """
        使用ImageNet标准化参数进行标准化
        公式：normalized = (pixel - mean) / std
        """
        # 确保mean和std在正确的设备上
        mean, std = device_manager.move_tensors_to_device(
            self.imagenet_mean, self.imagenet_std, device=image.device
        )
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)

        # 标准化
        image = (image - mean) / std

        return image


# ProcessedDataLoader 类已被移除，直接使用普通的 DataLoader


# 主要对外接口
def create_unified_dataloader(dataloader: DataLoader,
                              dataset_name: str,
                              **kwargs) -> DataLoader:
    """
    将原始DataLoader加工为统一格式的DataLoader
    
    这是一个加工器函数，输入DataLoader，输出经过统一处理的DataLoader。
    所有图像将被加工为 [3, 224, 224] RGB格式，并应用ImageNet标准化。
    
    加工步骤：
    1. 通道转换（灰度->RGB, RGBA->RGB）
    2. 按最短边缩放（使用双三次插值）
    3. 中心裁剪到224x224
    4. ImageNet标准化
    
    Args:
        dataloader: 原始 DataLoader
        dataset_name: 数据集名称
        **kwargs: 额外参数（如target_size等）
        
    Returns:
        DataLoader: 加工后的统一格式DataLoader
    
    Example:
        >>> unified_loader = create_unified_dataloader(mnist_loader, 'mnist')
        >>> for images, labels in unified_loader:
        ...     print(f"Images: {images.shape}")  # [batch_size, 3, 224, 224]
        ...     print(f"Image range: [{images.min():.2f}, {images.max():.2f}]")  # 约[-2.0, +2.0]
    """
    # 创建统一格式的数据集
    unified_dataset = UnifiedDataset(
        original_dataset=dataloader.dataset,
        dataset_name=dataset_name,
        **kwargs
    )

    # 直接创建并返回普通的DataLoader
    return DataLoader(
        unified_dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,  # 保持原有顺序
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        collate_fn=dataloader.collate_fn
    )
