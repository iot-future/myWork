"""
实验运行器模块
负责联邦学习实验的设置、运行和结果保存
"""

import os
import random
import numpy as np
import torch
from typing import Dict, Any, List
from torch.utils.data import DataLoader, ConcatDataset

from core.server import FederatedServer
from core.client import FederatedClient
from data.data_loader import get_client_dataloaders
from data.datasets.mnist import MNIST
from aggregation.federated_avg import FederatedAveraging
from utils.model_factory import ModelFactory
from utils.results_handler import ResultsHandler
from utils.wandb_logger import init_wandb, log_client_metrics, log_global_metrics, finish_wandb
from utils.dataset_stats import count_clients_per_dataset

class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.server = None
        self.clients = []
        self.test_loader = None
        self.use_wandb = config.get('wandb', {}).get('enabled', False)
        self.device = None
        self.dataset_client_counts, self.dataset_client_mappings = count_clients_per_dataset(config)

    def setup_environment(self):
        """设置实验环境"""
        # 设置设备
        device_config = self.config.get('device', 'auto')
        if device_config == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_config)
        
        # 设置随机种子
        seed = self.config['experiment']['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # 确保CUDA操作的确定性
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # 设置CUBLAS环境变量以支持确定性算法
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # 设置环境变量以确保完全确定性
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # 固定PyTorch的随机数生成器状态
        torch.use_deterministic_algorithms(True)
        
        print(f"✓ 环境设置完成: 随机种子 {seed}, 设备 {self.device}")
    
    def setup_data(self):
        """设置数据加载器"""
        data_config = self.config['data']
        client_config = self.config['client']
        
        # 获取配置参数
        num_clients = client_config['num_clients']
        batch_size = data_config['batch_size']
        data_root = data_config.get('data_dir', './data')
        
        # 准备基础数据集配置
        base_dataset_configs = {}
        for dataset_name, dataset_config in data_config['datasets'].items():
            base_dataset_configs[dataset_name] = {
                'data_root': data_root,
                **dataset_config
            }
        
        # 为每个客户端创建数据加载器
        client_data_loaders = []
        client_datasets_config = client_config['client_datasets']
        
        for client_id in range(num_clients):
            client_key = f"client_{client_id}"
            
            # 获取此客户端应该使用的数据集
            client_dataset_names = client_datasets_config[client_key]
            
            # 为此客户端准备数据集配置
            client_dataset_configs = {
                name: base_dataset_configs[name] 
                for name in client_dataset_names 
                if name in base_dataset_configs
            }
            
            # 获取此客户端的数据加载器
            client_dataloaders_dict = get_client_dataloaders(
                client_original_id=client_key,
                dataset_client_mappings=self.dataset_client_mappings,
                dataset_client_counts=self.dataset_client_counts,
                batch_size=batch_size,
                dataset_configs=client_dataset_configs,
                seed=self.config['experiment']['seed']
            )
            
            # 将数据加载器字典存储起来
            client_data_loaders.append(client_dataloaders_dict)
        
        # 创建测试数据加载器（使用第一个数据集作为测试集）
        first_dataset_name = list(base_dataset_configs.keys())[0]
        test_config = base_dataset_configs[first_dataset_name].copy()
        test_config['train'] = False
        
        from data.data_loader import SUPPORTED_DATASETS
        test_dataset = SUPPORTED_DATASETS[first_dataset_name](**test_config)
        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        # 计算并打印数据统计信息
        self._print_data_statistics(client_data_loaders, num_clients)
        
        return client_data_loaders
    
    def _print_data_statistics(self, client_data_loaders, num_clients):
        """打印数据统计信息"""
        print(f"✓ 数据加载完成: {num_clients} 个客户端")
        
        for client_id, dataloaders_dict in enumerate(client_data_loaders):
            dataset_info = []
            total_client_samples = 0
            
            for dataset_name, dataloader in dataloaders_dict.items():
                samples = len(dataloader.dataset)
                total_client_samples += samples
                dataset_info.append(f"{dataset_name}({samples})")
            
            datasets_str = ", ".join(dataset_info)
            print(f"  客户端 {client_id}: {datasets_str} - 总计 {total_client_samples} 样本")
    
    def setup_server(self):
        """设置服务器"""
        # 创建全局模型（使用配置化优化器）
        optimizer_config = self.config.get('optimizer', {})
        global_model = ModelFactory.create_model(
            self.config['model'], 
            optimizer_config if optimizer_config else None
        )
        
        # 将模型移到设备
        if hasattr(global_model, 'model'):
            global_model.model = global_model.model.to(self.device)
        
        # 创建聚合器
        aggregator = FederatedAveraging()
        
        # 创建服务器
        self.server = FederatedServer(global_model, aggregator)
        print("✓ 服务器设置完成")
    
    def setup_clients(self, client_data_loaders: List):
        """设置客户端"""
        client_config = self.config['client']
        optimizer_config = self.config.get('optimizer', {})
        
        self.clients = []
        for i, dataloaders_dict in enumerate(client_data_loaders):
            client_id = f"client_{i}"
            client_model = ModelFactory.create_model(
                self.config['model'], 
                optimizer_config if optimizer_config else None
            )
            
            # 将客户端模型移到设备
            if hasattr(client_model, 'model'):
                client_model.model = client_model.model.to(self.device)
            
            # 对于多数据集客户端，我们需要合并所有数据集的数据加载器
            # 这里我们选择第一个数据集作为主要数据集，或者可以实现数据集合并逻辑
            if len(dataloaders_dict) == 1:
                # 单数据集客户端
                data_loader = list(dataloaders_dict.values())[0]
            else:
                # 多数据集客户端：创建一个联合数据加载器
                data_loader = self._create_combined_dataloader(dataloaders_dict)
            
            client = FederatedClient(
                client_id=client_id,
                model=client_model,
                data_loader=data_loader,
                epochs=client_config['local_epochs'],
                learning_rate=client_config.get('learning_rate', optimizer_config.get('learning_rate', 0.01)),
                device=self.device
            )
            self.clients.append(client)
        
        print(f"✓ 客户端设置完成: {len(self.clients)} 个客户端")
    
    def _create_combined_dataloader(self, dataloaders_dict):
        """创建联合数据加载器，将多个数据集合并为一个数据加载器"""
        # 收集所有数据集
        datasets = []
        for dataset_name, dataloader in dataloaders_dict.items():
            datasets.append(dataloader.dataset)
        
        # 合并数据集
        combined_dataset = ConcatDataset(datasets)
        
        # 使用第一个数据加载器的配置参数
        first_loader = list(dataloaders_dict.values())[0]
        batch_size = first_loader.batch_size
        
        # 创建新的数据加载器
        combined_loader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        return combined_loader
    
    def run_federated_round(self, round_num: int) -> Dict[str, float]:
        """执行一轮联邦学习"""
        # 获取全局模型参数
        global_params = self.server.send_global_model()
        
        # 所有客户端进行本地训练
        client_updates = []
        for client in self.clients:
            client_result = client.train(global_params)
            client_updates.append(client_result)
            
            # 记录客户端指标到wandb
            if self.use_wandb and 'metrics' in client_result:
                metrics = client_result['metrics']
                log_client_metrics(
                    client.client_id, 
                    round_num, 
                    metrics.get('loss', 0.0),
                    metrics.get('accuracy')
                )
        
        # 服务器聚合
        self.server.aggregate(client_updates)
        
        # 评估
        metrics = {}
        if round_num % self.config['evaluation']['evaluate_every'] == 0:
            metrics = self.evaluate_global_model()
            progress_msg = ResultsHandler.format_training_progress(round_num, metrics)
            print(progress_msg)
            
            # 记录全局模型指标到wandb
            if self.use_wandb and metrics:
                log_global_metrics(round_num, metrics)
        
        return metrics
    
    def evaluate_global_model(self) -> Dict[str, float]:
        """评估全局模型"""
        if self.test_loader is None:
            return {}
        
        return self.server.evaluate_with_dataloader(self.test_loader)
    
    def run_experiment(self) -> List[Dict[str, Any]]:
        """运行完整实验"""
        print(f"开始实验: {self.config['experiment']['name']}")
        print("=" * 60)
        
        # 初始化wandb
        if self.use_wandb:
            project_name = self.config.get('wandb', {}).get('project', 'federated-learning')
            is_offline = self.config.get('wandb', {}).get('offline', False)
            init_wandb(self.config, project_name, is_offline)

        
        # 设置实验环境
        self.setup_environment()
        
        # 设置数据
        client_data_loaders = self.setup_data()
        
        # 设置服务器和客户端
        self.setup_server()
        self.setup_clients(client_data_loaders)
        
        print("\n开始联邦学习训练...")
        # 运行联邦学习轮次
        rounds = self.config['experiment']['rounds']
        all_metrics = []
        
        for round_num in range(1, rounds + 1):
            metrics = self.run_federated_round(round_num)
            if metrics:
                metrics['round'] = round_num
                all_metrics.append(metrics)
    
        
        print("=" * 60)
        print("✓ 实验完成!")
        
        # 结束wandb记录
        if self.use_wandb:
            finish_wandb()
        
        return all_metrics
