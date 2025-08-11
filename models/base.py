import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any
from core.base import BaseModel


class SimpleLinearModel(BaseModel):
    """简单线性模型实现"""
    
    def __init__(self, input_dim: int, output_dim: int, optimizer_config: Dict[str, Any] = None):
        """
        初始化线性模型
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            optimizer_config: 优化器配置字典
        """
        super().__init__(optimizer_config)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 创建简单的线性模型
        self.model = nn.Linear(input_dim, output_dim)
        
        # 使用配置化优化器或默认SGD
        self.create_optimizer(self.model.parameters())
        if self.optimizer is None:
            # 回退到默认SGD（保持向后兼容）
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        
        self.criterion = nn.MSELoss()
    
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
        
        # 前向传播
        outputs = self.model(data)
        loss = self.criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, data, labels):
        """模型评估"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            return {"loss": loss.item()}
    
    def evaluate_with_dataloader(self, dataloader):
        """使用数据加载器评估模型"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for data, labels in dataloader:
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                # 累计损失
                total_loss += loss.item() * data.size(0)
                total_samples += labels.size(0)
        
        avg_loss = total_loss / total_samples
        return {"loss": avg_loss}


class SimpleClassificationModel(BaseModel):
    """简单分类模型实现"""
    
    def __init__(self, input_dim: int, num_classes: int, learning_rate: float = 0.01):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        # 创建简单的分类模型
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
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
        
        # 前向传播
        outputs = self.model(data)
        loss = self.criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, data, labels):
        """模型评估"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            accuracy = correct / total
            
            return {"loss": loss.item(), "accuracy": accuracy}
    
    def evaluate_with_dataloader(self, dataloader):
        """使用数据加载器评估模型"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, labels in dataloader:
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                # 累计损失
                total_loss += loss.item() * data.size(0)
                
                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        return {"loss": avg_loss, "accuracy": accuracy}
