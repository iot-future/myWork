"""
data 模块的核心接口。

该文件提供了为联邦学习客户端创建数据加载器的主要功能。
它抽象了数据分区、子集创建和 DataLoader 实例化的复杂性。

设计原则：
- 数据集类只负责数据的加载和预处理变换
- DataLoader 的创建统一在此模块中进行，确保联邦学习的数据分区需求
- 支持灵活的批处理大小和工作进程配置
"""
from typing import Any, Dict, List, Tuple, Union
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from data.datasets import mnist, cifar10, cifar100
from data.middleware import create_unified_dataloader

# 支持的数据集名称到其对应类的映射
SUPPORTED_DATASETS = {
    'mnist': mnist.MNIST,
    'cifar10': cifar10.CIFAR10,
    'cifar100': cifar100.CIFAR100
}


def _validate_dataset_name(dataset_name: str) -> str:
    """验证数据集名称是否受支持。
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        str: 标准化的数据集名称（小写）
        
    Raises:
        NotImplementedError: 当数据集不受支持时
    """
    normalized_name = dataset_name.lower()
    if normalized_name not in SUPPORTED_DATASETS:
        supported_names = list(SUPPORTED_DATASETS.keys())
        raise NotImplementedError(
            f"不支持数据集 '{dataset_name}'。支持的数据集: {supported_names}"
        )
    return normalized_name


def _calculate_client_data_range(client_id: int, total_clients: int, total_items: int) -> Tuple[int, int]:
    """计算指定客户端应该获得的数据索引范围。
    
    Args:
        client_id: 客户端ID（从0开始）
        total_clients: 客户端总数
        total_items: 数据项总数
        
    Returns:
        Tuple[int, int]: (开始索引, 结束索引)
    """
    items_per_client = total_items // total_clients
    start_idx = client_id * items_per_client
    
    # 最后一个客户端获得剩余的所有数据
    end_idx = total_items if client_id == total_clients - 1 else start_idx + items_per_client
    
    return start_idx, end_idx


def _generate_shuffled_indices(dataset_size: int, seed: int) -> List[int]:
    """生成打乱的数据集索引。
    
    Args:
        dataset_size: 数据集大小
        seed: 随机种子
        
    Returns:
        List[int]: 打乱后的索引列表
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randperm(dataset_size, generator=generator).tolist()


def _split_dataset(
    dataset: Dataset, 
    total_clients: int, 
    client_dataset_id: int = None, 
    seed: int = 42
) -> Union[List[int], List[List[int]]]:
    """将数据集索引均匀（IID）地分配给客户端。
    
    Args:
        dataset: 要分割的数据集
        total_clients: 使用此数据集的客户端总数
        client_dataset_id: 客户端在此数据集中的ID (0 到 total_clients-1)，
                          如果指定，只返回该客户端的索引
        seed: 随机种子，用于确保可复现性
        
    Returns:
        Union[List[int], List[List[int]]]: 
            如果指定了client_dataset_id，返回该客户端的索引列表
            否则返回所有客户端的索引列表的列表
    """
    dataset_size = len(dataset)
    shuffled_indices = _generate_shuffled_indices(dataset_size, seed)

    if client_dataset_id is not None:
        # 返回指定客户端的索引
        start_idx, end_idx = _calculate_client_data_range(
            client_dataset_id, total_clients, dataset_size
        )
        return shuffled_indices[start_idx:end_idx]
    
    # 返回所有客户端的索引
    all_client_indices = []
    for client_id in range(total_clients):
        start_idx, end_idx = _calculate_client_data_range(
            client_id, total_clients, dataset_size
        )
        client_indices = shuffled_indices[start_idx:end_idx]
        all_client_indices.append(client_indices)
    
    return all_client_indices


def _create_single_client_dataloader(
    dataset_name: str,
    dataset_config: Dict[str, Any],
    client_original_id: str,
    dataset_client_mappings: Dict[str, Dict[str, int]],
    dataset_client_counts: Dict[str, int],
    batch_size: int,
    num_workers: int,
    seed: int
) -> DataLoader:
    """为单个数据集创建客户端的 DataLoader。
    
    Args:
        dataset_name: 数据集名称
        dataset_config: 数据集配置参数
        client_original_id: 客户端原始ID
        dataset_client_mappings: 数据集客户端映射字典
        dataset_client_counts: 每个数据集的客户端总数
        batch_size: 批处理大小
        num_workers: 工作进程数
        seed: 随机种子
        
    Returns:
        DataLoader: 为该客户端创建的数据加载器
    """
    # 1. 验证数据集名称
    normalized_name = _validate_dataset_name(dataset_name)
    
    # 2. 获取客户端在数据集内的ID
    client_internal_id = dataset_client_mappings[normalized_name][client_original_id]
    
    # 3. 创建完整数据集
    dataset_class = SUPPORTED_DATASETS[normalized_name]
    full_dataset = dataset_class(**dataset_config)

    # 4. 获取客户端的数据索引
    client_indices = _split_dataset(
        dataset=full_dataset, 
        total_clients=dataset_client_counts[normalized_name], 
        client_dataset_id=client_internal_id, 
        seed=seed
    )

    # 5. 创建客户端数据子集
    client_subset = Subset(full_dataset, client_indices)

    # 6. 创建 DataLoader
    dataloader = DataLoader(
        dataset=client_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # 7. 使用中间件统一处理
    return create_unified_dataloader(dataloader, normalized_name)


def get_client_dataloaders(
    client_original_id: str,
    dataset_client_mappings: Dict[str, Dict[str, int]],
    dataset_client_counts: Dict[str, int],
    batch_size: int,
    dataset_configs: Dict[str, Dict[str, Any]],
    num_workers: int = 0,
    seed: int = 42
) -> Dict[str, DataLoader]:
    """
    为特定客户端创建并返回一个或多个 DataLoader。

    每个客户端都会收到每个请求数据集的非重叠子集。

    Args:
        client_original_id: 客户端的原始ID（如 "client_0"）
        dataset_client_mappings: 数据集客户端映射字典
            {数据集名称: {客户端原始ID: 数据集内ID}}
        dataset_client_counts: 每个数据集的客户端总数
            {数据集名称: 客户端数量}
        batch_size: DataLoader 的批处理大小
        dataset_configs: 数据集配置字典，键是数据集名称，值是配置参数
            示例: {'mnist': {'data_root': './data/MNIST', 'train': True}}
        num_workers: 数据加载时使用的工作进程数，默认为0
        seed: 随机种子，用于数据集分割的可复现性，默认为42

    Returns:
        Dict[str, DataLoader]: 将数据集名称映射到相应 DataLoader 的字典
    """
    # 初始化客户端数据加载器字典，用于存储该客户端的所有数据集的DataLoader
    client_dataloaders = {}

    # 遍历每个请求的数据集配置
    for dataset_name, config in dataset_configs.items():
        # 为当前数据集创建专属于该客户端的DataLoader
        # 注意：每次调用都会复制配置以避免修改原始配置
        dataloader = _create_single_client_dataloader(
            dataset_name=dataset_name,  # 数据集名称（如 'mnist', 'cifar10'）
            dataset_config=config.copy(),  # 数据集配置的副本，防止意外修改
            client_original_id=client_original_id,  # 客户端的全局唯一ID
            dataset_client_mappings=dataset_client_mappings,  # 客户端在各数据集中的映射关系
            dataset_client_counts=dataset_client_counts,  # 各数据集的客户端总数
            batch_size=batch_size,  # 批处理大小
            num_workers=num_workers,  # 数据加载进程数
            seed=seed  # 随机种子，确保数据分割的可复现性
        )

        # 将创建的DataLoader存储到结果字典中
        # 键为数据集名称，值为对应的DataLoader实例
        client_dataloaders[dataset_name] = dataloader

    # 返回包含所有数据集DataLoader的字典
    return client_dataloaders


def _clean_dataset_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """清理数据集配置，移除不属于数据集构造函数的参数。
    
    Args:
        config: 原始配置字典
        
    Returns:
        Dict[str, Any]: 清理后的配置字典
    """
    cleaned_config = config.copy()
    # 移除 DataLoader 特定的参数
    dataloader_params = ['batch_size', 'num_workers']
    for param in dataloader_params:
        cleaned_config.pop(param, None)
    return cleaned_config


def get_dataset_info(dataset_name: str, dataset_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    获取数据集的基本信息，如类别名称、数据集大小等。

    Args:
        dataset_name: 数据集名称
        dataset_config: 数据集配置参数

    Returns:
        Dict[str, Any]: 包含数据集信息的字典，包括：
            - classnames: 类别名称列表
            - num_classes: 类别数量
            - dataset_size: 数据集大小
    """
    # 验证数据集名称
    normalized_name = _validate_dataset_name(dataset_name)
    
    # 清理配置参数
    clean_config = _clean_dataset_config(dataset_config)
    
    # 创建数据集实例
    dataset_class = SUPPORTED_DATASETS[normalized_name]
    dataset = dataset_class(**clean_config)

    return {
        'classnames': dataset.classnames,
        'num_classes': len(dataset.classnames),
        'dataset_size': len(dataset)
    }
