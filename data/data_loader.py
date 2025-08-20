"""
data 模块的核心接口。

该文件提供了为联邦学习客户端创建数据加载器的主要功能。
它抽象了数据分区、子集创建和 DataLoader 实例化的复杂性。

设计原则：
- 数据集类只负责数据的加载和预处理变换
- DataLoader 的创建统一在此模块中进行，确保联邦学习的数据分区需求
- 支持灵活的批处理大小和工作进程配置
"""
from typing import Dict, List
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from data.datasets import mnist, cifar10

# 支持的数据集名称到其对应类的映射
SUPPORTED_DATASETS = {
    'mnist': mnist.MNIST,
    'cifar10': cifar10.CIFAR10
}


def _split_dataset(dataset: Dataset, num_clients: int, client_id: int = None, seed: int = 42) -> List[List[int]]:
    """将数据集索引均匀（IID）地分配给客户端。
    
    Args:
        dataset: 要分割的数据集
        num_clients: 客户端总数
        client_id: 如果指定，只返回该客户端的索引
        seed: 随机种子，用于确保可复现性
    """
    num_items = len(dataset)
    items_per_client = num_items // num_clients
    
    # 为确保可复现性，在打乱前固定随机种子
    g = torch.Generator()
    g.manual_seed(seed)
    shuffled_indices = torch.randperm(num_items, generator=g).tolist()

    if client_id is not None:
        # 只返回指定客户端的索引
        start_idx = client_id * items_per_client
        if client_id == num_clients - 1:
            end_idx = num_items
        else:
            end_idx = start_idx + items_per_client
        return shuffled_indices[start_idx:end_idx]
    else:
        # 返回所有客户端的索引
        client_indices: List[List[int]] = [[] for _ in range(num_clients)]
        for client_id in range(num_clients):
            start_idx = client_id * items_per_client
            if client_id == num_clients - 1:
                end_idx = num_items
            else:
                end_idx = start_idx + items_per_client
            client_indices[client_id] = shuffled_indices[start_idx:end_idx]
        return client_indices


def get_client_dataloaders(
    client_id: int,
    num_clients: int,
    batch_size: int,
    dataset_configs: Dict[str, Dict],
    num_workers: int = 0,
    seed: int = 42
) -> Dict[str, DataLoader]:
    """
    为特定客户端创建并返回一个或多个 DataLoader。

    每个客户端都会收到每个请求数据集的非重叠子集。

    Args:
        client_id (int): 客户端 ID (从 0 到 num_clients-1)。
        num_clients (int): 客户端总数。
        batch_size (int): DataLoader 的批处理大小。
        dataset_configs (Dict[str, Dict]): 一个字典，键是数据集名称，
            值是该数据集构造函数的参数字典。
            示例: {'mnist': {'data_root': './data/MNIST', 'train': True}}
        num_workers (int): 数据加载时使用的工作进程数，默认为0。
        seed (int): 随机种子，用于数据集分割的可复现性，默认为42。

    Returns:
        Dict[str, DataLoader]: 一个将数据集名称映射到相应 DataLoader 的字典。
    """
    if not 0 <= client_id < num_clients:
        raise ValueError(f"客户端 ID 必须在 0 到 {num_clients-1} 之间")

    client_dataloaders = {}

    for name, config in dataset_configs.items():
        name_lower = name.lower()
        if name_lower not in SUPPORTED_DATASETS:
            supported_names = list(SUPPORTED_DATASETS.keys())
            raise NotImplementedError(
                f"不支持数据集 '{name}'。支持的数据集: {supported_names}"
            )

        # 1. 准备数据集配置，确保不传递DataLoader特定的参数给数据集
        dataset_config = config.copy()

        # 2. 加载完整数据集
        dataset_class = SUPPORTED_DATASETS[name_lower]
        full_dataset = dataset_class(**dataset_config)

        # 3. 为所有客户端拆分数据
        client_indices = _split_dataset(full_dataset, num_clients, client_id, seed)

        # 4. 为此客户端创建子集
        client_dataset = Subset(full_dataset, client_indices)

        # 5. 创建 DataLoader，使用统一的参数
        dataloader = DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        client_dataloaders[name] = dataloader

    return client_dataloaders


def get_dataset_info(dataset_name: str, dataset_config: Dict) -> Dict[str, any]:
    """
    获取数据集的基本信息，如类别名称、数据集大小等。

    Args:
        dataset_name (str): 数据集名称。
        dataset_config (Dict): 数据集配置参数。

    Returns:
        Dict[str, any]: 包含数据集信息的字典，包括：
            - classnames: 类别名称列表
            - num_classes: 类别数量
            - dataset_size: 数据集大小
    """
    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower not in SUPPORTED_DATASETS:
        supported_names = list(SUPPORTED_DATASETS.keys())
        raise NotImplementedError(
            f"不支持数据集 '{dataset_name}'。支持的数据集: {supported_names}"
        )

    # 创建数据集实例来获取信息
    dataset_config = dataset_config.copy()
    dataset_config.pop('batch_size', None)
    dataset_config.pop('num_workers', None)
    
    dataset_class = SUPPORTED_DATASETS[dataset_name_lower]
    dataset = dataset_class(**dataset_config)

    return {
        'classnames': dataset.classnames,
        'num_classes': len(dataset.classnames),
        'dataset_size': len(dataset)
    }
