import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any
from core.base import BaseModel


class CNNModel(BaseModel):
    """CNN模型实现 - 适配224x224 RGB图像输入
    基于联邦学习经典架构，适配更大尺寸图像
    
    模型架构:
    - Conv2D: 5x5, 32 channels, ReLU, MaxPool2D
    - Conv2D: 5x5, 64 channels, ReLU, MaxPool2D  
    - Conv2D: 5x5, 128 channels, ReLU, MaxPool2D
    - Conv2D: 5x5, 256 channels, ReLU, MaxPool2D
    - Dense: 512 units, ReLU
    - Dense: 10 units (output)
    - 输入形状: [batch_size, 3, 224, 224]
    """
    
    def __init__(self, optimizer_config: Dict[str, Any] = None):
        """
        初始化CNN模型
        
        Args:
            optimizer_config: 优化器配置字典
        """
        super().__init__(optimizer_config)
        
        # 创建CNN模型结构 - 适配224x224 RGB输入
        self.model = nn.Sequential(
            # 第一个卷积层: 5x5卷积，32通道，输入3通道RGB
            nn.Conv2d(3, 32, kernel_size=5),  # 输入: 3x224x224, 输出: 32x220x220
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出: 32x110x110
            
            # 第二个卷积层: 5x5卷积，64通道  
            nn.Conv2d(32, 64, kernel_size=5),  # 输出: 64x106x106
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出: 64x53x53
            
            # 第三个卷积层: 5x5卷积，128通道
            nn.Conv2d(64, 128, kernel_size=5),  # 输出: 128x49x49
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出: 128x24x24
            
            # 第四个卷积层: 5x5卷积，256通道
            nn.Conv2d(128, 256, kernel_size=5),  # 输出: 256x20x20
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出: 256x10x10
            
            # 展平层
            nn.Flatten(),  # 输出: 256*10*10 = 25600
            
            # 全连接层: 512单元
            nn.Linear(256 * 10 * 10, 512),  
            nn.ReLU(),
            
            # 输出层: 10个类别
            nn.Linear(512, 10)
        )
        
        # 创建AdamW优化器
        self.create_optimizer(self.model.parameters())
        if self.optimizer is None:
            # 回退到默认AdamW（保持向后兼容）
            from utils.optimizer_factory import OptimizerFactory
            default_config = OptimizerFactory.get_default_config()
            self.optimizer = OptimizerFactory.create_optimizer(
                self.model.parameters(), default_config
            )
        
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
        
        # 使用基类的统一设备管理
        data, labels = self._ensure_device_compatibility(data, labels)
        
        # 确保数据形状正确 (batch_size, 3, 224, 224)
        if len(data.shape) == 2:  # 如果是展平的数据
            # 假设展平的数据是 224*224*3 = 150528 维
            data = data.view(-1, 3, 224, 224)
        elif len(data.shape) == 3:  # 如果缺少通道维度
            if data.shape[1] == 224 and data.shape[2] == 224:
                # 假设是单通道，扩展为3通道
                data = data.unsqueeze(1).repeat(1, 3, 1, 1)
        
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
            # 使用基类的统一设备管理
            data, labels = self._ensure_device_compatibility(data, labels)
            
            # 确保数据形状正确 (batch_size, 3, 224, 224)
            if len(data.shape) == 2:  # 如果是展平的数据
                data = data.view(-1, 3, 224, 224)
            elif len(data.shape) == 3:  # 如果缺少通道维度
                if data.shape[1] == 224 and data.shape[2] == 224:
                    # 假设是单通道，扩展为3通道
                    data = data.unsqueeze(1).repeat(1, 3, 1, 1)
            
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            accuracy = correct / total if total > 0 else 0.0  # 统一使用小数格式
            
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
        
    def evaluate_with_dataloader(self, data_loader):
        """使用数据加载器评估模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in data_loader:
                # 使用基类的统一设备管理
                data, labels = self._ensure_device_compatibility(data, labels)
                
                # 确保数据形状正确 (batch_size, 3, 224, 224)
                if len(data.shape) == 2:  # 如果是展平的数据
                    data = data.view(-1, 3, 224, 224)
                elif len(data.shape) == 3:  # 如果缺少通道维度
                    if data.shape[1] == 224 and data.shape[2] == 224:
                        # 假设是单通道，扩展为3通道
                        data = data.unsqueeze(1).repeat(1, 3, 1, 1)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                # 修正：使用样本数加权平均
                total_loss += loss.item() * data.size(0)
                
                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # 统一使用小数格式和样本数加权平均
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def predict(self, data):
        """预测函数"""
        self.model.eval()
        with torch.no_grad():
            # 确保数据形状正确 (batch_size, 3, 224, 224)
            if len(data.shape) == 2:  # 如果是展平的数据
                data = data.view(-1, 3, 224, 224)
            elif len(data.shape) == 3:  # 如果缺少通道维度
                if data.shape[1] == 224 and data.shape[2] == 224:
                    # 假设是单通道，扩展为3通道
                    data = data.unsqueeze(1).repeat(1, 3, 1, 1)
            outputs = self.model(data)
            _, predicted = torch.max(outputs, 1)
            return predicted
