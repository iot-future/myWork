"""
数据集统计工具模块
用于计算每个数据集的客户端使用情况
"""

from typing import Dict, List, Any, Tuple


def count_clients_per_dataset(config: Dict[str, Any]) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    """
    计算使用每个数据集的客户端数量和在该数据集中的ID映射
    
    Args:
        config: 实验配置字典，包含客户端数据集配置
        
    Returns:
        Tuple[Dict[str, int], Dict[str, Dict[str, int]]]: 
        - 第一个返回值：每个数据集对应的客户端数量 {数据集名称: 客户端数量}
        - 第二个返回值：每个数据集的客户端映射 {数据集名称: {原始客户端ID: 数据集内ID}}
    """
    # 初始化结果字典
    dataset_client_counts = {}
    dataset_client_mappings = {}
    
    # 获取客户端数据集配置
    client_datasets = config.get('client', {}).get('client_datasets', {})
    
    # 获取所有可用数据集
    available_datasets = config.get('data', {}).get('datasets', {}).keys()
    
    # 初始化每个数据集的统计信息
    for dataset_name in available_datasets:
        dataset_client_counts[dataset_name] = 0
        dataset_client_mappings[dataset_name] = {}
    
    # 遍历每个客户端的数据集配置
    for client_id, datasets in client_datasets.items():
        if isinstance(datasets, list):
            # 对于当前客户端使用的每个数据集，分配数据集内的ID
            for dataset_name in datasets:
                if dataset_name in dataset_client_counts:
                    # 为该客户端在此数据集中分配一个从0开始的连续ID
                    dataset_internal_id = dataset_client_counts[dataset_name]
                    dataset_client_mappings[dataset_name][client_id] = dataset_internal_id
                    dataset_client_counts[dataset_name] += 1
    
    return dataset_client_counts, dataset_client_mappings

if __name__ == "__main__":
    """主函数 - 简洁的实验流程"""
    # 创建配置管理器
    from config_manager import ConfigManager
 
    config_manager = ConfigManager()
    
    # 解析命令行参数
    parser = config_manager.create_parser()
    args = parser.parse_args()
    
    # 加载配置并应用命令行覆盖
    config = config_manager.load_config(args.config)
    config = config_manager.override_config(config, args)
    
    # 获取数据集统计信息
    dataset_client_counts, dataset_client_mappings = count_clients_per_dataset(config=config)
    
    # 输出所有数据集的统计信息
    print("数据集统计信息:")
    for dataset_name in dataset_client_counts:
        if dataset_client_counts[dataset_name] > 0:
            print(f"\n{dataset_name} 数据集:")
            print(f"  客户端数量: {dataset_client_counts[dataset_name]}")
            print(f"  客户端映射: {dataset_client_mappings[dataset_name]}")
            # 输出详细映射信息
            for original_id, dataset_internal_id in dataset_client_mappings[dataset_name].items():
                print(f"    {original_id} -> 数据集内ID: {dataset_internal_id}")
