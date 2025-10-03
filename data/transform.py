import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from utils.device_manager import device_manager

class UnifiedTransform:
    """
    统一格式的图像转换类，适用于CIFAR100等数据集
    
    对图像进行实时加工，输出标准格式：
    - 图像：[3, 224, 224] RGB 张量，ImageNet标准化
    """

    def __init__(self, target_size: int = 224):
        """
        Args:
            target_size: 目标图像尺寸
        """
        self.target_size = target_size
        
        # ImageNet预训练模型的标准化参数
        self.imagenet_mean = torch.tensor([0.485, 0.458, 0.408])
        self.imagenet_std = torch.tensor([0.269, 0.261, 0.276])
        
        # 基础转换：将PIL图像转换为张量
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image):
        """处理单个图像"""
        # 转换为张量
        if not isinstance(image, torch.Tensor):
            image = self.to_tensor(image)
        
        # 加工图像
        image = self._process_image(image)
        return image

    def _process_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        按照指定逻辑加工图像：
        1. 处理通道数（灰度->RGB, RGBA->RGB）
        2. 按最短边缩放到合适尺寸
        3. 中心裁剪到目标尺寸
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
        按最短边缩放并中心裁剪到目标尺寸
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

        # 中心裁剪到目标尺寸
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
        mean = self.imagenet_mean.view(3, 1, 1)
        std = self.imagenet_std.view(3, 1, 1)

        # 标准化
        image = (image - mean) / std

        return image