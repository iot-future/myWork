"""
CLIP模型专用数据变换类
提供图像预处理和批处理功能，适配Hugging Face CLIP模型
支持联邦学习框架，提供设备管理和错误处理

主要改进：
1. 增加了设备管理（CPU/GPU自动选择）
2. 增强了错误处理和参数验证
3. 改进了数据类型处理（使用np.clip防止溢出）
4. 添加了日志记录功能
5. 为联邦学习场景添加了create_federated_dataloaders方法
6. 增加了配置信息获取功能
7. 优化了批处理性能
8. 增强了代码的鲁棒性和可维护性
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Union, Tuple, Optional, Dict, Any
from transformers import CLIPProcessor
import torchvision.transforms as transforms
import warnings
import logging

# 设置日志
logger = logging.getLogger(__name__)


class CLIPDataTransforms:
    """CLIP模型数据变换工具类"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", 
                 cache_dir: Optional[str] = None,
                 device: Optional[str] = None):
        """
        初始化CLIP数据变换
        
        Args:
            model_name: CLIP模型名称
            cache_dir: 缓存目录
            device: 设备类型 ('cpu', 'cuda', 'auto' 等)
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        # 设备管理
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device is None:
            self.device = "cpu"
        else:
            self.device = device
            
        try:
            self.processor = CLIPProcessor.from_pretrained(
                model_name, 
                cache_dir=cache_dir
            )
            logger.info(f"Initialized CLIP processor with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize CLIP processor: {e}")
            raise
        
        # 创建基础的图像变换（用于numpy数组转PIL图像）
        self.to_pil_transform = transforms.ToPILImage()
        self.to_tensor_transform = transforms.ToTensor()
    
    @staticmethod
    def numpy_to_pil(data: np.ndarray) -> List[Image.Image]:
        """
        将numpy数组转换为PIL图像列表
        
        Args:
            data: 输入数据，形状可能是:
                - (batch_size, height, width) 灰度图像
                - (batch_size, height, width, channels) 彩色图像
                - (batch_size, channels, height, width) PyTorch格式
                
        Returns:
            PIL图像列表
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a numpy array")
            
        # 数据类型和范围检查
        if data.dtype not in [np.uint8, np.float32, np.float64]:
            logger.warning(f"Unexpected data type: {data.dtype}, converting to float32")
            data = data.astype(np.float32)
        
        if len(data.shape) == 2:
            # 单个灰度图像 (height, width)
            if data.dtype != np.uint8:
                data = np.clip(data * 255, 0, 255).astype(np.uint8)
            return [Image.fromarray(data, mode='L')]
        
        elif len(data.shape) == 3:
            if data.shape[0] <= 3:
                # 单个图像，PyTorch格式 (channels, height, width)
                data = np.transpose(data, (1, 2, 0))
                if data.shape[2] == 1:
                    # 灰度图像
                    if data.dtype != np.uint8:
                        data = np.clip(data * 255, 0, 255).astype(np.uint8)
                    return [Image.fromarray(data.squeeze(), mode='L')]
                else:
                    # 彩色图像
                    if data.dtype != np.uint8:
                        data = np.clip(data * 255, 0, 255).astype(np.uint8)
                    return [Image.fromarray(data, mode='RGB')]
            else:
                # 单个灰度图像 (height, width)
                if data.dtype != np.uint8:
                    data = np.clip(data * 255, 0, 255).astype(np.uint8)
                return [Image.fromarray(data, mode='L')]
        
        elif len(data.shape) == 4:
            images = []
            if data.shape[1] <= 3:
                # PyTorch格式批次 (batch_size, channels, height, width)
                for i in range(data.shape[0]):
                    img = np.transpose(data[i], (1, 2, 0))
                    if img.shape[2] == 1:
                        # 灰度图像
                        if img.dtype != np.uint8:
                            img = np.clip(img * 255, 0, 255).astype(np.uint8)
                        images.append(Image.fromarray(img.squeeze(), mode='L'))
                    else:
                        # 彩色图像
                        if img.dtype != np.uint8:
                            img = np.clip(img * 255, 0, 255).astype(np.uint8)
                        images.append(Image.fromarray(img, mode='RGB'))
            else:
                # 标准格式批次 (batch_size, height, width, channels)
                for i in range(data.shape[0]):
                    img = data[i]
                    if len(img.shape) == 2:
                        # 灰度图像
                        if img.dtype != np.uint8:
                            img = np.clip(img * 255, 0, 255).astype(np.uint8)
                        images.append(Image.fromarray(img, mode='L'))
                    elif img.shape[2] == 1:
                        # 灰度图像
                        if img.dtype != np.uint8:
                            img = np.clip(img * 255, 0, 255).astype(np.uint8)
                        images.append(Image.fromarray(img.squeeze(), mode='L'))
                    else:
                        # 彩色图像
                        if img.dtype != np.uint8:
                            img = np.clip(img * 255, 0, 255).astype(np.uint8)
                        images.append(Image.fromarray(img, mode='RGB'))
            return images
        
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")
    
    def for_clip_model(self, data: Union[np.ndarray, torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        为CLIP模型准备数据
        
        Args:
            data: 输入数据，可以是numpy数组、torch张量或PIL图像列表
            
        Returns:
            预处理后的张量，适合CLIP模型输入
        """
        try:
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
            
            if isinstance(data, np.ndarray):
                # 转换为PIL图像
                pil_images = self.numpy_to_pil(data)
            elif isinstance(data, list) and all(isinstance(img, Image.Image) for img in data):
                pil_images = data
            else:
                raise ValueError("Data must be numpy array, torch tensor, or list of PIL Images")
            
            # 使用CLIP处理器预处理图像
            inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
            
            # 移动到指定设备
            pixel_values = inputs['pixel_values'].to(self.device)
            
            return pixel_values
            
        except Exception as e:
            logger.error(f"Error in for_clip_model: {e}")
            raise
    
    def for_mnist_clip(self, data: np.ndarray) -> torch.Tensor:
        """
        专门为MNIST数据适配CLIP模型的预处理
        
        Args:
            data: MNIST数据，形状为(batch_size, 784)或(batch_size, 28, 28)
            
        Returns:
            预处理后的张量
        """
        try:
            # 参数验证
            if not isinstance(data, np.ndarray):
                raise ValueError("Input must be numpy array")
            
            # 如果数据是展平的，重塑为图像格式
            if len(data.shape) == 2 and data.shape[1] == 784:
                data = data.reshape(-1, 28, 28)
            elif len(data.shape) == 2 and data.shape[1] != 784:
                raise ValueError(f"Expected 784 features for flattened MNIST, got {data.shape[1]}")
            elif len(data.shape) == 3 and data.shape[1:] != (28, 28):
                logger.warning(f"Non-standard MNIST shape: {data.shape}, expected (batch_size, 28, 28)")
            
            # 将单通道灰度图像转换为三通道RGB图像
            if len(data.shape) == 3:
                # 复制通道以创建RGB图像
                data = np.stack([data, data, data], axis=-1)  # (batch_size, 28, 28, 3)
            
            # 调用通用的CLIP预处理
            return self.for_clip_model(data)
            
        except Exception as e:
            logger.error(f"Error in for_mnist_clip: {e}")
            raise
    
    def for_cifar_clip(self, data: np.ndarray) -> torch.Tensor:
        """
        专门为CIFAR数据适配CLIP模型的预处理
        
        Args:
            data: CIFAR数据，形状为(batch_size, 32, 32, 3)或(batch_size, 3, 32, 32)
            
        Returns:
            预处理后的张量
        """
        try:
            # 参数验证
            if not isinstance(data, np.ndarray):
                raise ValueError("Input must be numpy array")
                
            # 检查数据形状
            if len(data.shape) == 4:
                if data.shape[1:] == (3, 32, 32):
                    # PyTorch格式，转换为标准格式
                    logger.info("Converting from PyTorch format (batch, channels, height, width)")
                    data = np.transpose(data, (0, 2, 3, 1))
                elif data.shape[1:] != (32, 32, 3):
                    logger.warning(f"Non-standard CIFAR shape: {data.shape}, expected (batch_size, 32, 32, 3)")
            else:
                raise ValueError(f"Expected 4D array for CIFAR data, got shape: {data.shape}")
            
            return self.for_clip_model(data)
            
        except Exception as e:
            logger.error(f"Error in for_cifar_clip: {e}")
            raise
    
    def batch_preprocess(self, data_batch: Union[np.ndarray, torch.Tensor], 
                        batch_size: int = 32,
                        show_progress: bool = False) -> List[torch.Tensor]:
        """
        批量预处理数据
        
        Args:
            data_batch: 输入数据批次
            batch_size: 批处理大小
            show_progress: 是否显示进度信息
            
        Returns:
            预处理后的张量列表
        """
        try:
            if isinstance(data_batch, torch.Tensor):
                data_batch = data_batch.cpu().numpy()
            
            if not isinstance(data_batch, np.ndarray):
                raise ValueError("Input must be numpy array or torch tensor")
            
            total_samples = data_batch.shape[0]
            processed_batches = []
            
            if show_progress:
                logger.info(f"Processing {total_samples} samples in batches of {batch_size}")
            
            for i in range(0, total_samples, batch_size):
                end_idx = min(i + batch_size, total_samples)
                batch_data = data_batch[i:end_idx]
                
                if show_progress:
                    logger.info(f"Processing batch {i//batch_size + 1}/{(total_samples-1)//batch_size + 1}")
                
                processed_batch = self.for_clip_model(batch_data)
                processed_batches.append(processed_batch)
            
            return processed_batches
            
        except Exception as e:
            logger.error(f"Error in batch_preprocess: {e}")
            raise
    
    def get_config(self) -> Dict[str, Any]:
        """
        获取当前配置信息
        
        Returns:
            配置信息字典
        """
        return {
            'model_name': self.model_name,
            'cache_dir': self.cache_dir,
            'device': self.device,
            'processor_type': type(self.processor).__name__
        }


class CLIPDataLoader:
    """
    CLIP模型专用数据加载器
    结合CLIPDataTransforms提供完整的数据加载和预处理功能
    支持联邦学习场景的分布式数据处理
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 cache_dir: Optional[str] = None,
                 device: Optional[str] = None):
        """
        初始化CLIP数据加载器
        
        Args:
            model_name: CLIP模型名称
            cache_dir: 缓存目录
            device: 设备类型
        """
        self.transforms = CLIPDataTransforms(model_name, cache_dir, device)
        self.model_name = model_name
        self.device = self.transforms.device
        
        logger.info(f"Initialized CLIP data loader on device: {self.device}")
    
    def prepare_training_data(self, images: Union[np.ndarray, torch.Tensor], 
                            labels: Union[np.ndarray, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        准备训练数据
        
        Args:
            images: 图像数据
            labels: 标签数据
            
        Returns:
            预处理后的图像张量和标签张量
        """
        try:
            # 参数验证
            if images is None or labels is None:
                raise ValueError("Images and labels cannot be None")
                
            # 预处理图像
            processed_images = self.transforms.for_clip_model(images)
            
            # 确保标签是张量格式
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels).long()
            elif isinstance(labels, list):
                labels = torch.tensor(labels, dtype=torch.long)
            elif not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long)
            
            # 移动到相同设备
            labels = labels.to(self.device)
            
            # 验证数据维度匹配
            if processed_images.shape[0] != labels.shape[0]:
                raise ValueError(f"Batch size mismatch: images {processed_images.shape[0]}, labels {labels.shape[0]}")
            
            logger.info(f"Prepared data: images {processed_images.shape}, labels {labels.shape}")
            return processed_images, labels
            
        except Exception as e:
            logger.error(f"Error in prepare_training_data: {e}")
            raise
    
    def create_dataloader(self, images: Union[np.ndarray, torch.Tensor],
                         labels: Union[np.ndarray, torch.Tensor],
                         batch_size: int = 16,
                         shuffle: bool = True,
                         num_workers: int = 0,
                         pin_memory: Optional[bool] = None) -> torch.utils.data.DataLoader:
        """
        创建PyTorch数据加载器
        
        Args:
            images: 图像数据
            labels: 标签数据
            batch_size: 批处理大小
            shuffle: 是否打乱数据
            num_workers: 工作进程数量
            pin_memory: 是否使用固定内存
            
        Returns:
            PyTorch数据加载器
        """
        try:
            processed_images, processed_labels = self.prepare_training_data(images, labels)
            
            # 创建数据集
            dataset = torch.utils.data.TensorDataset(processed_images, processed_labels)
            
            # 自动设置pin_memory
            if pin_memory is None:
                pin_memory = torch.cuda.is_available() and self.device.startswith('cuda')
            
            # 创建数据加载器
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False  # 联邦学习中保留最后一个批次很重要
            )
            
            logger.info(f"Created DataLoader: {len(dataset)} samples, batch_size={batch_size}")
            return dataloader
            
        except Exception as e:
            logger.error(f"Error in create_dataloader: {e}")
            raise
    
    def create_federated_dataloaders(self, 
                                   images: Union[np.ndarray, torch.Tensor],
                                   labels: Union[np.ndarray, torch.Tensor],
                                   num_clients: int = 10,
                                   batch_size: int = 16,
                                   shuffle: bool = True,
                                   distribution: str = "uniform") -> List[torch.utils.data.DataLoader]:
        """
        为联邦学习创建多个客户端数据加载器
        
        Args:
            images: 图像数据
            labels: 标签数据
            num_clients: 客户端数量
            batch_size: 批处理大小
            shuffle: 是否打乱数据
            distribution: 数据分布方式 ('uniform', 'random')
            
        Returns:
            客户端数据加载器列表
        """
        try:
            total_samples = len(images) if hasattr(images, '__len__') else images.shape[0]
            
            if distribution == "uniform":
                # 均匀分配数据
                samples_per_client = total_samples // num_clients
                dataloaders = []
                
                for i in range(num_clients):
                    start_idx = i * samples_per_client
                    end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else total_samples
                    
                    client_images = images[start_idx:end_idx]
                    client_labels = labels[start_idx:end_idx]
                    
                    dataloader = self.create_dataloader(
                        client_images, client_labels, batch_size, shuffle
                    )
                    dataloaders.append(dataloader)
                    
                logger.info(f"Created {num_clients} federated dataloaders with uniform distribution")
                return dataloaders
                
            else:
                raise NotImplementedError(f"Distribution method '{distribution}' not implemented")
                
        except Exception as e:
            logger.error(f"Error in create_federated_dataloaders: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取数据加载器统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'transforms_config': self.transforms.get_config(),
            'device': self.device,
            'model_name': self.model_name
        }


# 使用示例和测试
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=== CLIP数据变换测试 ===")
    
    try:
        # 创建变换器
        transforms = CLIPDataTransforms(device="auto")
        print(f"设备: {transforms.device}")
        
        # 测试MNIST数据
        print("\n测试MNIST数据变换:")
        mnist_data = np.random.rand(4, 784).astype(np.float32)  # 模拟4个MNIST样本
        processed_mnist = transforms.for_mnist_clip(mnist_data)
        print(f"MNIST数据形状: {mnist_data.shape} -> {processed_mnist.shape}")
        
        # 测试CIFAR数据
        print("\n测试CIFAR数据变换:")
        cifar_data = np.random.rand(4, 32, 32, 3).astype(np.float32)
        processed_cifar = transforms.for_cifar_clip(cifar_data)
        print(f"CIFAR数据形状: {cifar_data.shape} -> {processed_cifar.shape}")
        
        # 测试数据加载器
        print("\n测试数据加载器:")
        loader = CLIPDataLoader(device="auto")
        
        # 模拟数据
        images = np.random.rand(100, 28, 28).astype(np.float32)
        labels = np.random.randint(0, 10, 100)
        
        # 创建普通数据加载器
        dataloader = loader.create_dataloader(images, labels, batch_size=16)
        
        for batch_idx, (batch_images, batch_labels) in enumerate(dataloader):
            print(f"批次 {batch_idx}: 图像形状 {batch_images.shape}, 标签形状 {batch_labels.shape}")
            if batch_idx >= 2:  # 只显示前3个批次
                break
        
        # 测试联邦学习数据加载器
        print("\n测试联邦学习数据加载器:")
        federated_loaders = loader.create_federated_dataloaders(
            images, labels, num_clients=3, batch_size=8
        )
        
        for client_id, client_loader in enumerate(federated_loaders):
            print(f"客户端 {client_id}: 数据集大小 {len(client_loader.dataset)}")
        
        # 测试配置信息
        print("\n数据加载器配置:")
        stats = loader.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        logger.error(f"测试失败: {e}")
        print(f"测试失败: {e}")
    
    print("\n=== CLIP数据变换测试完成 ===")
