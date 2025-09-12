import torch
from typing import Dict, Any
import copy
from .base import BaseClient
from utils.device_manager import device_manager


class FederatedClient(BaseClient):
    """联邦学习客户端实现"""
    
    def __init__(self, client_id: str, model, data_loader=None, epochs=1, learning_rate=0.01, device=None):
        super().__init__(client_id)
        self.model = model
        self.data_loader = data_loader
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device or torch.device('cpu')
    
    def train(self, global_model_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        本地训练实现
        返回模型参数和训练指标
        """
        if global_model_params:
            self.model.set_parameters(global_model_params)
        
        if self.data_loader is None:
            raise ValueError("Data loader not set")
        
        # 执行本地训练并收集训练指标
        total_loss = 0.0
        total_samples = 0
        
        for epoch in range(self.epochs):
            for batch_data, batch_labels in self.data_loader:
                # 将数据移到设备
                batch_data, batch_labels = device_manager.move_tensors_to_device(
                    batch_data, batch_labels, device=self.device
                )
                
                loss = self.model.train_step(batch_data, batch_labels)
                total_loss += loss * batch_data.size(0)
                total_samples += batch_data.size(0)
        
        # 计算平均损失
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        # 评估训练后的模型（获取准确率）
        eval_metrics = self.evaluate_on_local_data()
        
        return {
            'parameters': self.model.get_parameters(),
            'metrics': {
                'loss': avg_loss,
                'accuracy': eval_metrics.get('accuracy')
            }
        }
    
    def evaluate_on_local_data(self) -> Dict[str, float]:
        """在本地数据上评估模型"""
        if self.data_loader is None:
            return {}
        
        try:
            return self.model.evaluate_with_dataloader(self.data_loader)
        except Exception as e:
            print(f"⚠️  客户端 {self.client_id} 本地数据评估失败: {str(e)}")
            return {}
    
    def set_data(self, data_loader):
        """设置数据加载器"""
        self.data_loader = data_loader
    
    def evaluate(self, test_data, test_labels):
        """评估客户端模型"""
        try:
            return self.model.evaluate(test_data, test_labels)
        except Exception as e:
            print(f"⚠️  客户端 {self.client_id} 模型评估失败: {str(e)}")
            return {}
