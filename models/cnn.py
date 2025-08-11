import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any
from core.base import BaseModel


class CNNModel(BaseModel):
    """MNIST数据集的CNN模型实现 - 严格按照联邦学习开山论文架构
    Communication-Efficient Learning of Deep Networks from Decentralized Data
    McMahan et al., 2017
    
    模型架构:
    - Conv2D: 5x5, 32 channels, ReLU, MaxPool2D
    - Conv2D: 5x5, 64 channels, ReLU, MaxPool2D  
    - Dense: 512 units, ReLU
    - Dense: 10 units (output)
    - 总参数数量: 1,663,370
    """
    
    def __init__(self, optimizer_config: Dict[str, Any] = None):
        """
        初始化CNN模型
        
        Args:
            optimizer_config: 优化器配置字典
        """
        super().__init__(optimizer_config)
        
        # 创建CNN模型结构 - 严格按照论文
        self.model = nn.Sequential(
            # 第一个卷积层: 5x5卷积，32通道
            nn.Conv2d(1, 32, kernel_size=5),  # 输入: 1x28x28, 输出: 32x24x24
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出: 32x12x12
            
            # 第二个卷积层: 5x5卷积，64通道  
            nn.Conv2d(32, 64, kernel_size=5),  # 输出: 64x8x8
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出: 64x4x4
            
            # 展平层
            nn.Flatten(),  # 输出: 64*4*4 = 1024
            
            # 全连接层: 512单元（按照论文）
            nn.Linear(64 * 4 * 4, 512),  
            nn.ReLU(),
            
            # 输出层: 10个类别
            nn.Linear(512, 10)
        )
        
        # 使用配置化优化器或默认SGD
        self.create_optimizer(self.model.parameters())
        if self.optimizer is None:
            # 回退到默认SGD（保持向后兼容）
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        
        self.criterion = nn.CrossEntropyLoss()
        
    
    def get_parameters(self) -> Dict[str, Any]:
        """获取模型参数"""
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
    
    def set_parameters(self, params: Dict[str, Any]):
        """设置模型参数"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params:
                    param.data.copy_(params[name])
    
    def train_step(self, data, labels):
        """单步训练"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # 确保数据形状正确 (batch_size, 1, 28, 28)
        if len(data.shape) == 2:  # 如果是展平的数据
            data = data.view(-1, 1, 28, 28)
        
        # 前向传播
        outputs = self.model(data)
        loss = self.criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, data, labels):
        """模型评估 - 符合基类接口"""
        self.model.eval()
        with torch.no_grad():
            # 将数据移到模型所在设备
            device = next(self.model.parameters()).device
            data = data.to(device)
            labels = labels.to(device)
            
            # 确保数据形状正确 (batch_size, 1, 28, 28)
            if len(data.shape) == 2:  # 如果是展平的数据
                data = data.view(-1, 1, 28, 28)
            
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            accuracy = 100 * correct / total if total > 0 else 0
            
            return {
                'loss': loss.item(),
                'accuracy': accuracy
            }
    
    def evaluate_with_dataloader(self, data_loader):
        """使用数据加载器评估模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in data_loader:
                # 将数据移到模型所在设备
                device = next(self.model.parameters()).device
                data = data.to(device)
                labels = labels.to(device)
                
                # 确保数据形状正确 (batch_size, 1, 28, 28)
                if len(data.shape) == 2:  # 如果是展平的数据
                    data = data.view(-1, 1, 28, 28)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total if total > 0 else 0
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def predict(self, data):
        """预测函数"""
        self.model.eval()
        with torch.no_grad():
            if len(data.shape) == 2:  # 如果是展平的数据
                data = data.view(-1, 1, 28, 28)
            outputs = self.model(data)
            _, predicted = torch.max(outputs, 1)
            return predicted
