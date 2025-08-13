"""
演示模型模板 - 可直接复制修改使用

这是一个完整的模型实现示例，展示了如何正确地继承BaseModel
并实现所有必要的方法。您可以复制此文件并修改为自己的模型。

使用步骤：
1. 复制此文件到 models/ 目录
2. 重命名类名和文件名
3. 修改模型架构
4. 更新模型工厂
5. 创建配置文件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from core.base import BaseModel


class DemoModel(BaseModel):
    """
    演示模型类 - 实现了一个简单的多层感知机
    
    这个模型展示了如何正确实现BaseModel的所有抽象方法，
    以及一些最佳实践。您可以将其作为模板使用。
    
    架构：
    - 输入层 -> 隐藏层1 -> 隐藏层2 -> 输出层
    - 使用ReLU激活函数
    - 支持Dropout
    """
    
    def __init__(self, 
                 input_dim: int = 784,
                 hidden_dims: list = [256, 128],
                 output_dim: int = 10,
                 dropout_rate: float = 0.2,
                 activation: str = 'relu',
                 optimizer_config: Optional[Dict[str, Any]] = None):
        """
        初始化演示模型
        
        Args:
            input_dim: 输入维度，默认784（28x28 MNIST图像展平）
            hidden_dims: 隐藏层维度列表，默认[256, 128]
            output_dim: 输出维度，默认10（10个类别）
            dropout_rate: Dropout比例，默认0.2
            activation: 激活函数类型，默认'relu'
            optimizer_config: 优化器配置字典
        """
        # 1. 调用父类构造函数（必须）
        super().__init__(optimizer_config)
        
        # 2. 保存模型参数
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        
        # 3. 构建模型架构
        self.model = self._build_network()
        
        # 4. 创建AdamW优化器（必须调用）
        self.create_optimizer(self.model.parameters())
        if self.optimizer is None:
            # 使用默认AdamW配置作为后备
            from utils.optimizer_factory import OptimizerFactory
            default_config = OptimizerFactory.get_default_config()
            self.optimizer = OptimizerFactory.create_optimizer(
                self.model.parameters(), default_config
            )
        
        # 5. 定义损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 6. 初始化权重（可选）
        self._init_weights()
        
        print(f"✓ 创建DemoModel: {self._count_parameters():,} 参数")
    
    def _build_network(self) -> nn.Module:
        """构建网络架构"""
        layers = []
        
        # 输入层到第一个隐藏层
        prev_dim = self.input_dim
        
        # 添加隐藏层
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        return nn.Sequential(*layers)
    
    def _get_activation(self) -> nn.Module:
        """根据配置返回激活函数"""
        activation_map = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        return activation_map.get(self.activation.lower(), nn.ReLU())
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _count_parameters(self) -> int:
        """计算模型参数总数"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    # ========== BaseModel 抽象方法实现（必须） ==========
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        获取模型参数 - 联邦学习核心功能
        
        Returns:
            Dict[str, torch.Tensor]: 参数名称到参数张量的映射
        """
        return {
            name: param.data.clone() 
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
    
    def set_parameters(self, params: Dict[str, torch.Tensor]):
        """
        设置模型参数 - 联邦学习核心功能
        
        Args:
            params: 参数名称到参数张量的映射
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params and param.requires_grad:
                    param.data.copy_(params[name])
    
    def train_step(self, data: torch.Tensor, labels: torch.Tensor) -> float:
        """
        执行一步训练
        
        Args:
            data: 输入数据，形状为 [batch_size, input_dim]
            labels: 标签，形状为 [batch_size]
            
        Returns:
            float: 训练损失值
        """
        # 设置为训练模式
        self.model.train()
        
        # 清零梯度
        self.optimizer.zero_grad()
        
        # 前向传播
        outputs = self.model(data)
        loss = self.criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪（可选，防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # 更新参数
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, data: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            data: 输入数据，形状为 [batch_size, input_dim]  
            labels: 标签，形状为 [batch_size]
            
        Returns:
            Dict[str, float]: 包含损失和准确率的字典
        """
        # 设置为评估模式
        self.model.eval()
        
        with torch.no_grad():
            # 前向传播
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            accuracy = correct / total
            
            # 计算Top-5准确率（如果类别数>=5）
            top5_accuracy = None
            if self.output_dim >= 5:
                _, top5_pred = outputs.topk(5, 1, largest=True, sorted=True)
                top5_correct = top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
                top5_accuracy = top5_correct / total
        
        result = {
            'loss': loss.item(),
            'accuracy': accuracy
        }
        
        if top5_accuracy is not None:
            result['top5_accuracy'] = top5_accuracy
            
        return result
    
    # ========== 额外的实用方法 ==========
    
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """
        进行预测
        
        Args:
            data: 输入数据
            
        Returns:
            torch.Tensor: 预测结果（类别索引）
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data)
            _, predicted = torch.max(outputs, 1)
        return predicted
    
    def predict_proba(self, data: torch.Tensor) -> torch.Tensor:
        """
        预测概率分布
        
        Args:
            data: 输入数据
            
        Returns:
            torch.Tensor: 类别概率分布
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data)
            probabilities = F.softmax(outputs, dim=1)
        return probabilities
    
    def get_feature_representation(self, data: torch.Tensor, layer_index: int = -2) -> torch.Tensor:
        """
        获取中间层特征表示
        
        Args:
            data: 输入数据
            layer_index: 层索引，-2表示倒数第二层
            
        Returns:
            torch.Tensor: 特征表示
        """
        self.model.eval()
        with torch.no_grad():
            x = data
            for i, layer in enumerate(self.model):
                x = layer(x)
                if i == len(self.model) + layer_index:
                    return x
        return x
    
    def save_model(self, filepath: str):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'input_dim': self.input_dim,
                'hidden_dims': self.hidden_dims,
                'output_dim': self.output_dim,
                'dropout_rate': self.dropout_rate,
                'activation': self.activation
            }
        }, filepath)
    
    def load_model(self, filepath: str):
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
        """
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        获取模型摘要信息
        
        Returns:
            Dict[str, Any]: 模型摘要
        """
        return {
            'model_name': 'DemoModel',
            'total_parameters': self._count_parameters(),
            'architecture': {
                'input_dim': self.input_dim,
                'hidden_dims': self.hidden_dims,
                'output_dim': self.output_dim,
                'dropout_rate': self.dropout_rate,
                'activation': self.activation
            },
            'optimizer': type(self.optimizer).__name__,
            'criterion': type(self.criterion).__name__
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"DemoModel({self.input_dim} -> {' -> '.join(map(str, self.hidden_dims))} -> {self.output_dim})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"DemoModel(input_dim={self.input_dim}, "
                f"hidden_dims={self.hidden_dims}, "
                f"output_dim={self.output_dim}, "
                f"dropout_rate={self.dropout_rate}, "
                f"activation='{self.activation}')")


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 这个部分展示了如何使用DemoModel
    print("=== DemoModel 使用示例 ===")
    
    # 1. 创建模型
    model = DemoModel(
        input_dim=784,
        hidden_dims=[512, 256, 128],
        output_dim=10,
        dropout_rate=0.3,
        activation='relu'
    )
    
    print(f"模型摘要: {model.get_model_summary()}")
    
    # 2. 模拟数据
    batch_size = 32
    data = torch.randn(batch_size, 784)
    labels = torch.randint(0, 10, (batch_size,))
    
    # 3. 训练步骤
    loss = model.train_step(data, labels)
    print(f"训练损失: {loss:.4f}")
    
    # 4. 评估
    metrics = model.evaluate(data, labels)
    print(f"评估结果: {metrics}")
    
    # 5. 预测
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)
    print(f"预测形状: {predictions.shape}, 概率形状: {probabilities.shape}")
    
    # 6. 联邦学习参数操作
    params = model.get_parameters()
    print(f"参数数量: {len(params)}")
    
    model.set_parameters(params)
    print("✓ 参数设置成功")
    
    print("\n✅ 所有功能测试通过！")
