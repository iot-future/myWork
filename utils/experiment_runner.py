"""
实验运行器模块
负责联邦学习实验的设置、运行和结果保存
"""

import os
import random
import numpy as np
import torch
import time
import sys
from typing import Dict, Any, List
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from core.server import FederatedServer
from core.client import FederatedClient
from aggregation.federated_avg import FederatedAveraging
from utils.model_factory import ModelFactory
from utils.results_handler import ResultsHandler
from utils.wandb_logger import init_wandb, log_client_metrics, log_global_metrics, finish_wandb
from utils.dataset_stats import count_clients_per_dataset
from utils.evaluation_manager import EvaluationManager
from data.data_loader import get_client_dataloaders, create_test_loaders, create_combined_dataloader
from data.middleware import create_unified_dataloader
from utils.device_manager import device_manager


class ExperimentRunner:
    """
    联邦学习实验运行器
    
    Attributes:
        config (Dict[str, Any]): 实验配置字典，包含所有实验参数
        server (FederatedServer): 联邦学习服务器实例
        clients (List[FederatedClient]): 联邦学习客户端列表
        test_loaders (Dict[str, DataLoader]): 各数据集的测试数据加载器
        use_wandb (bool): 是否使用 Weights & Biases 记录实验
        device (torch.device): 训练设备（CPU/GPU）
        dataset_client_counts (Dict): 各数据集的客户端数量统计
        dataset_client_mappings (Dict): 数据集到客户端的映射关系
    
    Example:
        >>> config = {
        ...     'experiment': {'name': 'fed_avg_mnist', 'rounds': 10, 'seed': 42},
        ...     'model': {'name': 'cnn', 'num_classes': 10},
        ...     'client': {'num_clients': 5, 'local_epochs': 3},
        ...     'data': {'datasets': {'mnist': {}}, 'batch_size': 32}
        ... }
        >>> runner = ExperimentRunner(config)
        >>> results = runner.run_experiment()
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.server = None
        self.clients = []
        self.test_loaders = {}
        self.use_wandb = config.get('wandb', {}).get('enabled', False)
        self.device = None
        self.dataset_client_counts, self.dataset_client_mappings = count_clients_per_dataset(config)

        # 创建评估管理器
        verbose = config.get('evaluation', {}).get('verbose', True)
        self.evaluation_manager = EvaluationManager(verbose=verbose)

        # 时间追踪变量
        self.experiment_start_time = None
        self.round_times = []
        self.client_training_times = []

    def setup_environment(self):
        """设置实验环境"""
        # 设置设备
        device_config = self.config.get('device', 'auto')
        self.device = device_manager.get_optimal_device(device_config)

        # 设置随机种子
        seed = self.config['experiment']['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if self.device.type == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.use_deterministic_algorithms(True)

        print(f"✓ 环境设置完成: 随机种子 {seed}, 设备 {self.device}")

    def setup_data(self):
        """设置数据加载器"""
        data_config = self.config['data']  # 数据配置
        client_config = self.config['client']  # 客户端配置

        # 获取配置参数
        num_clients = client_config['num_clients']
        batch_size = data_config['batch_size']
        data_root = data_config.get('data_dir', './data')

        # 获取每个数据集的路径等基本信息
        base_dataset_configs = {
            dataset_name: {'data_root': data_root} if dataset_config is None else {'data_root': data_root,
                                                                                   **dataset_config}
            for dataset_name, dataset_config in data_config['datasets'].items()
        }

        # 为每个客户端创建数据加载器
        client_data_loaders = []
        client_datasets_config = client_config['client_datasets']

        for client_id in range(num_clients):
            client_key = f"client_{client_id}"
            # 获取该客户端应该使用的数据集列表
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

            client_data_loaders.append(client_dataloaders_dict)

        # 创建多数据集测试数据加载器
        self.test_loaders = create_test_loaders(base_dataset_configs, batch_size)

        # 计算并打印数据统计信息
        self._print_data_statistics(client_data_loaders, num_clients)

        # 打印测试集信息
        if self.test_loaders:
            print(f"\n✓ 测试数据集 ({len(self.test_loaders)} 个):")
        else:
            print("\n⚠️  无可用测试数据集")

        return client_data_loaders


    def _prepare_test_config(self, dataset_config: Dict[str, Any]) -> Dict[str, Any]:
        """准备测试数据集配置"""
        test_config = dataset_config.copy()
        test_config['train'] = False
        return test_config

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

    def setup_server(self, dataset_names: List[str]):
        """设置服务器"""
        optimizer_config = self.config.get('optimizer', {})
        global_model = ModelFactory.create_model(
            self.config['model'],
            optimizer_config if optimizer_config else None,
            dataset_names=dataset_names,
            device=self.device
        )

        # 将模型移到设备
        global_model = device_manager.move_model_to_device(global_model, self.device)

        # 添加LoRA状态验证
        if hasattr(global_model, 'is_lora_enabled') and global_model.is_lora_enabled():
            lora_info = global_model.get_lora_info()
            print(f"🎯 全局模型LoRA状态: 已启用 | 可训练参数: {lora_info.get('trainable_parameters', 0):,}")
        elif hasattr(global_model, 'is_lora_enabled'):
            print("📸 全局模型: 正常训练")

        # 创建聚合器和服务器
        aggregator = FederatedAveraging()
        self.server = FederatedServer(global_model, aggregator)
        print("✓ 服务器设置完成")

    def setup_clients(self, client_data_loaders: List):
        """设置客户端"""
        client_config = self.config['client']
        optimizer_config = self.config.get('optimizer', {})

        print(f"\n🔧 开始设置 {len(client_data_loaders)} 个客户端...")

        self.clients = []
        lora_clients_count = 0

        for i, dataloaders_dict in enumerate(client_data_loaders):
            client_id = f"client_{i}"
            # 从数据加载器字典中提取数据集名称
            client_dataset_names = list(dataloaders_dict.keys())
            client_model = ModelFactory.create_model(
                self.config['model'],
                optimizer_config if optimizer_config else None,
                dataset_names=client_dataset_names,
                device=self.device
            )

            # 将客户端模型移到设备
            client_model = device_manager.move_model_to_device(client_model, self.device)

            # 统计LoRA启用的客户端数量
            if hasattr(client_model, 'is_lora_enabled') and client_model.is_lora_enabled():
                lora_clients_count += 1

            # 处理数据加载器：单数据集或多数据集
            data_loader = (list(dataloaders_dict.values())[0] if len(dataloaders_dict) == 1
               else create_combined_dataloader(dataloaders_dict))

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
        if lora_clients_count > 0:
            print(f"🎯 LoRA启用客户端: {lora_clients_count}/{len(self.clients)}")


    def run_federated_round(self, round_num: int) -> Dict[str, float]:
        """执行一轮联邦学习"""
        round_start_time = time.time()

        # 获取全局模型参数
        global_params = self.server.send_global_model()

        # 所有客户端进行本地训练 - 使用嵌套的进度条
        client_updates = []
        client_metrics = {}  # 用于收集每个客户端的评估结果

        # 检查是否显示详细进度（batch级别）
        show_batch_progress = self.config.get('training', {}).get('show_batch_progress', False)

        with tqdm(self.clients, desc=f"第{round_num}轮训练",
                  bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                  ncols=None, leave=False, position=1, file=sys.stdout) as pbar:

            for _, client in enumerate(pbar):
                # 传递show_progress参数以启用batch级别的进度条
                client_result = client.train(global_params, show_progress=show_batch_progress)
                client_updates.append(client_result)

                # 收集客户端评估指标
                metrics = client_result.get('metrics', {})
                client_metrics[client.client_id] = metrics
                
                # 打印客户端指标
                if show_batch_progress:
                    print(f"客户端 {client.client_id} 准确率: {metrics.get('accuracy', 0) * 100:.2f}%, 损失: {metrics.get('loss', 0):.4f}")

                # 简化的进度信息
                loss = client_result.get('metrics', {}).get('loss', 0)
                pbar.set_postfix({'Loss': f'{loss:.3f}'})

                # 刷新显示以避免重叠
                pbar.refresh()

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

        # 评估（使用新的评估管理器）
        metrics = {}
        if round_num % self.config['evaluation']['evaluate_every'] == 0:
            # 评估全局模型
            global_metrics = self.evaluation_manager.evaluate_global_model(
                self.server, self.test_loaders, round_num
            )

            # 创建评估结果
            eval_result = self.evaluation_manager.create_evaluation_result(
                round_num, client_metrics, global_metrics
            )

            # 打印轮次总结
            progress_msg = self.evaluation_manager.format_round_summary(eval_result)
            print(progress_msg)

            # 记录全局模型指标到wandb
            if self.use_wandb and global_metrics:
                log_global_metrics(round_num, global_metrics)

            metrics = global_metrics

        return metrics

    def run_experiment(self) -> List[Dict[str, Any]]:
        """运行完整实验"""
        print(f"开始实验: {self.config['experiment']['name']}")
        print("=" * 60)

        # 初始化wandb
        if self.use_wandb:
            project_name = self.config.get('experiment', {}).get('name', 'federated-learning')
            is_offline = self.config.get('wandb', {}).get('offline', False)
            init_wandb(self.config, project_name, is_offline)

        # 设置实验环境
        self.setup_environment()

        # 设置数据
        client_data_loaders = self.setup_data()

        # 设置服务器和客户端
        self.setup_server(list(self.test_loaders.keys()))
        self.setup_clients(client_data_loaders)

        print("\n开始联邦学习训练...")
        # 运行联邦学习轮次
        rounds = self.config['experiment']['rounds']
        all_metrics = []
        self.experiment_start_time = time.time()

        # 在训练开始前记录LoRA参数状态（如果启用）
        lora_enabled = hasattr(self.server.global_model,
                               'is_lora_enabled') and self.server.global_model.is_lora_enabled()
        if lora_enabled:
            initial_lora_params = self.server.global_model.get_parameters()
            lora_param_count = len([k for k in initial_lora_params.keys() if 'lora_' in k])
            print(f"🔄 LoRA训练模式: {lora_param_count} 个LoRA参数层将被优化")

        # 使用总体进度条
        with tqdm(range(1, rounds + 1), desc="实验进度", unit="轮",
                  position=0, file=sys.stdout, ncols=None) as round_pbar:
            for round_num in round_pbar:
                metrics = self.run_federated_round(round_num)
                if metrics:
                    metrics['round'] = round_num
                    all_metrics.append(metrics)

                # 更新总体进度条
                if metrics and 'accuracy' in metrics:
                    round_pbar.set_postfix({'Acc': f"{metrics['accuracy']:.2f}%"})

                # 在最后一轮验证LoRA参数更新
                if lora_enabled and round_num == rounds:
                    current_lora_params = self.server.global_model.get_parameters()
                    lora_keys = [k for k in current_lora_params.keys() if 'lora_' in k]
                    print(f"✅ LoRA训练完成: {len(lora_keys)} 个参数层已优化")

        # 打印最终总结
        self._print_final_summary()

        print("=" * 60)
        print("✅ 实验完成!")

        # 结束wandb记录
        if self.use_wandb:
            finish_wandb()

        return all_metrics

    def _print_final_summary(self):
        """打印最终实验总结"""
        summary = self.evaluation_manager.get_final_summary()
        if not summary:
            print("⚠️  没有可用的实验结果")
            return

        print(f"\n📊 实验总结:")
        print(f"总训练轮次: {summary['total_rounds']}")
        print(f"最终准确率: {summary['final_accuracy']:.2f}%")
        print(f"最终损失: {summary['final_loss']:.4f}")

        if summary['total_rounds'] > 1:
            print(f"准确率提升: +{summary['accuracy_improvement']:.2f}%")
            print(f"损失降低: -{summary['loss_reduction']:.4f}")

        print(f"参与客户端: {summary['client_count']} 个")
