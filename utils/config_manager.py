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
        'batch_size': ['data', 'batch_size']
    }

    @staticmethod
    def load_config(config_file: str = 'configs/default.yaml') -> Dict[str, Any]:
        '''加载YAML配置文件'''
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

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

        return parser
