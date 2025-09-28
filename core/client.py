import torch
from typing import Dict, Any
import copy
from .base import BaseClient
from utils.device_manager import device_manager
from tqdm import tqdm
import sys
from torch.utils.data import random_split, DataLoader, TensorDataset, Subset
from data.data_loader import GroupBatchSampler


def split_client_data_loader(data_loader, train_ratio=0.8):
    """
    将客户端数据划分为训练集和测试集
    Args:
        data_loader: 所有数据加载器
        train_ratio: 训练集比例
    Returns:
        train_loader, test_loader
    """
    dataset = data_loader.dataset
    # 如果数据集不是ConcatDataset，则直接使用random_split划分
    if not isinstance(dataset, torch.utils.data.ConcatDataset):
        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        test_size = total_size - train_size
        train_datasets, test_datasets = random_split(dataset=dataset,lengths=[train_size, test_size])
        train_loader = DataLoader(train_datasets, batch_size=data_loader.batch_size, shuffle=True, num_workers=data_loader.num_workers)
        test_loader = DataLoader(test_datasets, batch_size=data_loader.batch_size, shuffle=False, num_workers=data_loader.num_workers)
        return train_loader, test_loader
    
    # 如果数据集是ConcatDataset，则使用GroupBatchSampler
    train_sampler = GroupBatchSampler(dataset, data_loader.batch_size, train=True, train_ratio=train_ratio, shuffle=True)
    test_sampler = GroupBatchSampler(dataset, data_loader.batch_size, train=False, train_ratio=train_ratio, shuffle=False)
    train_loader = DataLoader(dataset, batch_sampler=train_sampler, num_workers=data_loader.num_workers)
    test_loader = DataLoader(dataset, batch_sampler=test_sampler, num_workers=data_loader.num_workers)
    
    return train_loader, test_loader


class FederatedClient(BaseClient):
    """联邦学习客户端实现

    Args: 
        client_id (str): 客户端的唯一标识符。 
        model: 客户端使用的模型。 
        data_loader: 所有数据加载器，默认为None。 
        epochs (int): 训练轮数，默认为1。 
        learning_rate (float): 学习率，默认为0.01。 
        device: 训练设备，默认为CPU。 
    """

    def __init__(self, client_id: str, model, data_loader=None, epochs=1, learning_rate=0.01, device=None):
        super().__init__(client_id)
        self.model = model
        # 划分训练集和测试集
        self.train_loader, self.test_loader = split_client_data_loader(data_loader, train_ratio=0.8)

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device or torch.device('cpu')

    def get_parameters(self) -> Dict[str, Any]:
        """获取模型参数"""
        return self.model.get_parameters()

    def train(self, global_model_params: Dict[str, Any], show_progress: bool = False) -> Dict[str, Any]:
        """
        本地训练实现
        返回模型参数和训练指标
        """
        if global_model_params:
            self.model.set_parameters(global_model_params)

        if self.train_loader is None:
            raise ValueError("Data loader not set")

        # 执行本地训练并收集训练指标
        total_loss = 0.0
        total_samples = 0

        # 计算总的batch数量
        total_batches = len(self.train_loader) * self.epochs

        # 创建batch级别的进度条
        if show_progress:
            batch_pbar = tqdm(
                total=total_batches,
                desc=f"  {self.client_id} 训练",
                unit="batch",
                leave=False,
                ncols=None,  # 自适应终端宽度
                position=2,
                file=sys.stdout,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )

        for epoch in range(self.epochs):
            for batch_data, batch_labels, dataset_names in self.train_loader:
                # 将数据移到设备
                batch_data, batch_labels = device_manager.move_tensors_to_device(
                    batch_data, batch_labels, device=self.device
                )

                loss = self.model.train_step(batch_data, batch_labels, dataset_names)
                total_loss += loss * batch_data.size(0)
                total_samples += batch_data.size(0)

                # 更新batch进度条
                if show_progress:
                    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
                    batch_pbar.set_postfix({
                        'Epoch': f'{epoch + 1}/{self.epochs}',
                        'Loss': f'{avg_loss:.4f}'
                    })
                    batch_pbar.update(1)
                    batch_pbar.refresh()  # 刷新显示

        # 关闭batch进度条
        if show_progress:
            batch_pbar.close()
            # 确保进度条完全清除
            import time
            time.sleep(0.01)

        # 计算平均损失
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        # 评估训练后的模型（获取准确率）
        eval_metrics = self.evaluate()

        return {
            'parameters': self.model.get_parameters(),
            'metrics': {
                'loss': avg_loss,
                'accuracy': eval_metrics.get('accuracy')
            }
        }

    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
        if self.test_loader is None:
            return {}

        try:
            return self.model.evaluate(self.test_loader)
        except Exception as e:
            print(f"⚠️  客户端 {self.client_id} 本地数据评估失败: {str(e)}")
            return {}
