import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from typing import List, Tuple, Callable, Optional
import os


# 数据变换函数
class DataTransforms:
    """数据变换工具类"""
    
    @staticmethod
    def for_linear_model(data: np.ndarray) -> torch.Tensor:
        """线性模型的数据变换：保持数据展平状态"""
        return torch.FloatTensor(data)
    
    @staticmethod
    def for_cnn_model(data: np.ndarray, channels: int = 1, height: int = 28, width: int = 28) -> torch.Tensor:
        """CNN模型的数据变换：将展平数据重新整形为图像格式"""
        if len(data.shape) == 2:
            # 假设数据已经展平，重新整形为图像格式
            return torch.FloatTensor(data).view(-1, channels, height, width)
        else:
            return torch.FloatTensor(data)
    
    @staticmethod
    def for_rnn_model(data: np.ndarray, sequence_length: int) -> torch.Tensor:
        """RNN模型的数据变换：将数据整形为序列格式"""
        if len(data.shape) == 2:
            batch_size, features = data.shape
            # 重新整形为序列格式 (batch_size, sequence_length, features_per_step)
            features_per_step = features // sequence_length
            return torch.FloatTensor(data).view(batch_size, sequence_length, features_per_step)
        else:
            return torch.FloatTensor(data)


class SimpleDataset(Dataset):
    """简单数据集类"""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, data_type: str = "classification", transform=None):
        """
        初始化数据集
        
        Args:
            data: 输入数据
            labels: 标签数据
            data_type: 数据类型 ("classification" 或 "regression")
            transform: 数据变换函数，用于将数据转换为模型期望的格式
        """
        # 应用数据变换
        if transform is not None:
            self.data = transform(data)
        else:
            self.data = torch.FloatTensor(data)
        
        # 分类任务使用LongTensor，回归任务使用FloatTensor
        if data_type == "classification":
            self.labels = torch.LongTensor(labels)
        else:
            self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class FederatedDataLoader:
    """联邦学习数据加载器"""
    
    def __init__(self, num_clients: int, batch_size: int = 32, data_transform: Optional[Callable] = None, data_root: str = "./data"):
        """
        初始化联邦数据加载器
        
        Args:
            num_clients: 客户端数量
            batch_size: 批次大小
            data_transform: 数据变换函数，用于将数据转换为模型期望的格式
            data_root: 数据存储根目录
        """
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.data_transform = data_transform or DataTransforms.for_linear_model
        self.data_root = data_root
        self.client_data = {}
    
    def load_mnist_dataset(self,
                          random_state: int = 42,
                          samples_per_client: Optional[int] = None,
                          data_root: Optional[str] = None) -> Tuple[List[DataLoader], DataLoader]:
        """
        加载MNIST数据集（支持IID数据分布）
        
        Args:
            random_state: 随机种子
            samples_per_client: 每个客户端的样本数量（用于基线实验）
            data_root: 数据根目录，如果为None则使用实例的data_root
            
        Returns:
            (客户端数据加载器列表, 测试数据加载器)
        """
        # 使用传入的data_root参数或实例的data_root
        dataset_root = data_root if data_root is not None else self.data_root
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        train_dataset = datasets.MNIST(dataset_root, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(dataset_root, train=False, download=True, transform=transform)
        
        # 提取训练数据
        X_train, y_train = self._extract_from_dataset(train_dataset)
        
        # 创建联邦数据分布
        if samples_per_client is not None:
            # 基线实验：每个客户端固定样本数
            client_dataloaders = self._create_baseline_federated_split(
                X_train, y_train, random_state, samples_per_client)
        else:
            # 标准IID分布：平均分配所有数据
            client_dataloaders = self._create_federated_split(X_train, y_train, random_state)
        
        # 创建测试数据加载器
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return client_dataloaders, test_dataloader
    
    def _extract_from_dataset(self, dataset) -> Tuple[np.ndarray, np.ndarray]:
        """从PyTorch数据集中提取特征和标签"""
        X_list = []
        y_list = []
        
        for data, label in dataset:
            if isinstance(data, torch.Tensor):
                # 将图像数据展平
                X_list.append(data.numpy().flatten())
            else:
                X_list.append(data)
            y_list.append(label)
        
        return np.array(X_list), np.array(y_list)
    
    def _create_baseline_federated_split(self, 
                                        X: np.ndarray, 
                                        y: np.ndarray, 
                                        random_state: int,
                                        samples_per_client: int) -> List[DataLoader]:
        """
        创建基线联邦数据分布（按照开山论文设置）
        每个客户端固定样本数的IID分布
        
        Args:
            X: 训练数据
            y: 训练标签  
            random_state: 随机种子
            samples_per_client: 每个客户端的样本数量
        """
        client_dataloaders = []
        total_samples = len(X)
        total_needed = self.num_clients * samples_per_client
        
        # 确保有足够的数据
        if total_needed > total_samples:
            print(f"警告: 需要 {total_needed} 个样本，但只有 {total_samples} 个样本")
            print(f"将使用所有可用数据，每个客户端约 {total_samples // self.num_clients} 个样本")
            return self._create_federated_split(X, y, random_state)
        
        # 随机打乱数据（IID设置）
        indices = np.random.RandomState(random_state).permutation(total_samples)
        
        # 为每个客户端分配固定数量的样本
        for i in range(self.num_clients):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client
            client_indices = indices[start_idx:end_idx]
            
            client_X = X[client_indices]
            client_y = y[client_indices]
            
            # 创建数据集和数据加载器
            dataset = SimpleDataset(client_X, client_y, "classification", self.data_transform)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            client_dataloaders.append(dataloader)
            
            # 存储客户端数据信息
            self.client_data[f"client_{i}"] = {
                "size": samples_per_client,
                "indices": client_indices.tolist()
            }
        
        print(f"成功创建基线联邦数据分布:")
        print(f"- 客户端数量: {self.num_clients}")
        print(f"- 每客户端样本数: {samples_per_client}")
        print(f"- 总使用样本数: {self.num_clients * samples_per_client}")
        print(f"- 数据分布: IID (随机打乱后分配)")
        
        return client_dataloaders

    def _create_federated_split(self, 
                               X: np.ndarray, 
                               y: np.ndarray, 
                               random_state: int) -> List[DataLoader]:
        """创建联邦数据分布（仅IID分布）"""
        client_dataloaders = []
        
        # IID分布：随机分配数据
        indices = np.random.RandomState(random_state).permutation(len(X))
        split_indices = np.array_split(indices, self.num_clients)
        
        # 为每个客户端创建数据加载器
        for i, client_indices in enumerate(split_indices):
            client_X = X[client_indices]
            client_y = y[client_indices]
            
            # 创建数据集和数据加载器
            dataset = SimpleDataset(client_X, client_y, "classification", self.data_transform)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            client_dataloaders.append(dataloader)
            
            # 存储客户端数据信息
            self.client_data[f"client_{i}"] = {
                "size": len(client_indices),
                "indices": client_indices.tolist()
            }
        
        return client_dataloaders
    
    def get_client_data_info(self) -> dict:
        """获取客户端数据信息"""
        return self.client_data
    
    def get_data_statistics(self, client_dataloaders: List[DataLoader]) -> dict:
        """获取数据统计信息"""
        stats = {
            "total_clients": len(client_dataloaders),
            "client_sizes": [],
            "total_samples": 0
        }
        
        for i, dataloader in enumerate(client_dataloaders):
            client_size = len(dataloader.dataset)
            stats["client_sizes"].append(client_size)
            stats["total_samples"] += client_size
        
        stats["avg_client_size"] = stats["total_samples"] / stats["total_clients"]
        stats["min_client_size"] = min(stats["client_sizes"])
        stats["max_client_size"] = max(stats["client_sizes"])
        
        return stats
