import argparse
import yaml
import os
from typing import Dict, Any


class ConfigManager:
    ARG_CONFIG_MAP = {
        'rounds': ['experiment', 'rounds'],
        'seed': ['experiment', 'seed'],
        'num_clients': ['client', 'num_clients'],
        'local_epochs': ['client', 'local_epochs'],
        'learning_rate': [['client', 'learning_rate'], ['model', 'learning_rate']],
        'batch_size': ['data', 'batch_size'],
        'data_dir': ['data', 'data_dir']
    }

    @staticmethod
    def load_config(config_file: str = 'configs/default.yaml') -> Dict[str, Any]:
        '''加载YAML配置文件'''
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 验证配置文件的完整性
        ConfigManager._validate_config(config)
        return config
    
    @staticmethod
    def _validate_config(config: Dict[str, Any]):
        """验证配置文件的完整性和正确性"""
        # 验证必需的配置项
        required_sections = ['experiment', 'client', 'data', 'model']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"配置文件缺少必需的节: {section}")
        
        # 验证必需的多数据集配置
        client_config = config.get('client', {})
        data_config = config.get('data', {})
        
        # 必须有 client_datasets 配置
        if 'client_datasets' not in client_config:
            raise ValueError(f"配置文件缺少必需的 client.client_datasets 配置")
        
        # 必须有 datasets 配置
        if 'datasets' not in data_config:
            raise ValueError(f"配置文件缺少必需的 data.datasets 配置")
        
        client_datasets = client_config['client_datasets']
        num_clients = client_config['num_clients']
        
        # 检查是否为所有客户端都配置了数据集
        for i in range(num_clients):
            client_key = f"client_{i}"
            if client_key not in client_datasets:
                raise ValueError(f"客户端 {client_key} 未在 client_datasets 中配置")
        
        # 检查配置的数据集是否在 data.datasets 中存在
        available_datasets = set(data_config['datasets'].keys())
        for client_key, datasets in client_datasets.items():
            if not datasets:
                raise ValueError(f"客户端 {client_key} 必须至少配置一个数据集")
            for dataset_name in datasets:
                if dataset_name not in available_datasets:
                    raise ValueError(f"客户端 {client_key} 配置的数据集 '{dataset_name}' 不在 data.datasets 中")
        
        print("✓ 配置文件验证通过")

    @staticmethod
    def override_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
        '''根据命令行参数覆盖配置'''
        args = ConfigManager._process_args(args)

        for arg_name, config_path in ConfigManager.ARG_CONFIG_MAP.items():
            value = getattr(args, arg_name, None)
            if value is not None:
                ConfigManager._set_nested_value(config, config_path, value)

        return config

    @staticmethod
    def _set_nested_value(config: Dict[str, Any], path, value):
        '''设置嵌套配置值'''
        if isinstance(path[0], list):  # 多个路径的情况
            for p in path:
                ConfigManager._set_single_path(config, p, value)
        else:
            ConfigManager._set_single_path(config, path, value)

    @staticmethod
    def _set_single_path(config: Dict[str, Any], path, value):
        '''设置单一路径的配置值'''
        current = config
        for key in path[:-1]:
            current = current.setdefault(key, {})
        current[path[-1]] = value

    @staticmethod
    def _process_args(args: argparse.Namespace) -> argparse.Namespace:
        '''处理命令行参数'''
        return args

    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description='联邦学习实验')

        parser.add_argument('--config', '-c', default='configs/default.yaml',
                            help='配置文件路径')
        parser.add_argument('--rounds', type=int, help='训练轮次')
        parser.add_argument('--seed', type=int, help='随机种子')
        parser.add_argument('--num-clients', type=int, help='客户端数量')
        parser.add_argument('--local-epochs', type=int, help='本地训练轮次')
        parser.add_argument('--learning-rate', type=float, help='学习率')
        parser.add_argument('--batch-size', type=int, help='批大小')
        parser.add_argument('--data-dir', type=str, help='数据存储目录')

        return parser
