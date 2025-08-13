# 联邦学习项目中新增模型的完整指南

本文档将详细介绍如何在联邦学习项目中一步步新增一个自定义模型。我们将以CLIP模型为例，但这个流程适用于任何新模型。

## 目录

1. [项目结构概述](#1-项目结构概述)
2. [步骤一：创建模型实现](#2-步骤一创建模型实现)
3. [步骤二：更新模型工厂](#3-步骤二更新模型工厂)
4. [步骤三：更新模块导出](#4-步骤三更新模块导出)
5. [步骤四：创建配置文件](#5-步骤四创建配置文件)

---

## 1. 项目结构概述

在开始之前，让我们了解项目的关键结构：

```
myWork/
├── models/                 # 模型定义目录
│   ├── __init__.py        # 模块导出文件
│   ├── base.py           # 基础线性模型
│   ├── cnn.py            # CNN模型
│   └── clip.py           # CLIP模型（示例）
├── utils/
│   └── model_factory.py  # 模型工厂类
├── configs/              # 配置文件目录
│   ├── mnist.yaml
│   └── clip.yaml
├── examples/             # 示例代码目录
└── docs/                 # 文档目录
```

关键文件说明：
- **`models/`目录**：存放所有模型的实现
- **`core/base.py`**：定义了`BaseModel`基类
- **`utils/model_factory.py`**：模型工厂，负责根据配置创建模型
- **`models/__init__.py`**：导出模型供其他模块使用

---

## 2. 步骤一：创建模型实现

### 2.1 理解BaseModel基类

首先，您需要理解所有模型都必须继承的`BaseModel`基类：

```python
# core/base.py 中的BaseModel
class BaseModel(ABC):
    def __init__(self, optimizer_config: Dict[str, Any] = None):
        self.optimizer_config = optimizer_config
        self.optimizer = None
        
    def create_optimizer(self, model_parameters):
        """创建优化器"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """获取模型参数 - 联邦学习必需"""
        pass
    
    @abstractmethod
    def set_parameters(self, params: Dict[str, Any]):
        """设置模型参数 - 联邦学习必需"""
        pass
    
    @abstractmethod
    def train_step(self, data, labels=None):
        """单步训练"""
        pass
    
    @abstractmethod
    def evaluate(self, data, labels=None):
        """评估模型"""
        pass
```

### 2.2 创建新模型文件

**步骤详解：**

1. 在`models/`目录下创建新文件，命名为`your_model_name.py`
2. 导入必要的依赖
3. 继承`BaseModel`类
4. 实现所有抽象方法

**具体实现模板：**

```python
# models/your_model_name.py

"""
您的自定义模型实现
描述模型的用途、架构特点等
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any
from core.base import BaseModel


class YourCustomModel(BaseModel):
    """您的自定义模型类"""
    
    def __init__(self, 
                 # 模型特有的参数
                 input_dim: int = 784,
                 hidden_dim: int = 128,
                 output_dim: int = 10,
                 # 通用参数
                 optimizer_config: Dict[str, Any] = None):
        """
        初始化模型
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            optimizer_config: 优化器配置
        """
        super().__init__(optimizer_config)
        
        # 1. 保存模型参数
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 2. 定义模型架构
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 3. 创建优化器（必须）
        self.create_optimizer(self.model.parameters())
        if self.optimizer is None:
            # 提供默认优化器作为后备
            # 使用默认AdamW优化器
            from utils.optimizer_factory import OptimizerFactory
            default_config = OptimizerFactory.get_default_config()
            self.optimizer = OptimizerFactory.create_optimizer(
                self.model.parameters(), default_config
            )
        
        # 4. 定义损失函数
        self.criterion = nn.CrossEntropyLoss()
    
    def get_parameters(self) -> Dict[str, Any]:
        """获取模型参数 - 联邦学习核心功能"""
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
    
    def set_parameters(self, params: Dict[str, Any]):
        """设置模型参数 - 联邦学习核心功能"""
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
        """评估模型"""
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            accuracy = correct / total
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy
        }
```

### 2.3 复杂模型的实现技巧

对于更复杂的模型（如CLIP），可以：

1. **分离子组件**：
```python
class VisionEncoder(nn.Module):
    """视觉编码器"""
    def __init__(self, ...):
        # 实现细节
        pass

class TextEncoder(nn.Module):
    """文本编码器"""
    def __init__(self, ...):
        # 实现细节
        pass

class CLIPModel(BaseModel):
    def __init__(self, ...):
        super().__init__(optimizer_config)
        
        self.vision_encoder = VisionEncoder(...)
        self.text_encoder = TextEncoder(...)
        
        # 将所有组件组合
        self.model = nn.ModuleDict({
            'vision_encoder': self.vision_encoder,
            'text_encoder': self.text_encoder
        })
```

2. **自定义前向传播**：
```python
def forward(self, images, text):
    """自定义前向传播逻辑"""
    image_features = self.vision_encoder(images)
    text_features = self.text_encoder(text)
    return image_features, text_features
```

---

## 3. 步骤二：更新模型工厂

模型工厂负责根据配置创建模型实例。您需要在`utils/model_factory.py`中添加新模型。

### 3.1 添加导入

首先在文件顶部添加新模型的导入：

```python
# utils/model_factory.py

from typing import Dict, Any
from models.cnn import CNNModel
from models.base import SimpleLinearModel
from models.clip import CLIPModel
from models.your_model_name import YourCustomModel  # 添加这行
```

### 3.2 更新create_model方法

在`create_model`方法中添加新的模型类型处理：

```python
@staticmethod
def create_model(model_config: Dict[str, Any], optimizer_config: Dict[str, Any] = None):
    """根据配置创建模型"""
    model_type = model_config['type']
    
    if model_type == 'cnn':
        return CNNModel(optimizer_config=optimizer_config)
    elif model_type == 'linear':
        # 现有代码...
        pass
    elif model_type == 'your_model':  # 添加新的模型类型
        return YourCustomModel(
            input_dim=model_config.get('input_dim', 784),
            hidden_dim=model_config.get('hidden_dim', 128),
            output_dim=model_config.get('output_dim', 10),
            optimizer_config=optimizer_config
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
```

### 3.3 更新支持的模型列表

```python
@staticmethod
def get_supported_models():
    """获取支持的模型类型列表"""
    return ['cnn', 'linear', 'clip', 'your_model']  # 添加新模型
```

---

## 4. 步骤三：更新模块导出

更新`models/__init__.py`文件以导出新模型：

```python
# models/__init__.py

"""
模型模块

包含联邦学习支持的各种深度学习模型
"""

from .base import SimpleLinearModel
from .cnn import CNNModel
from .clip import CLIPModel
from .your_model_name import YourCustomModel  # 添加这行

__all__ = [
    'SimpleLinearModel',
    'CNNModel', 
    'CLIPModel',
    'YourCustomModel'  # 添加这行
]
```

---

## 5. 步骤四：创建配置文件

### 5.1 理解配置文件结构

配置文件采用YAML格式，包含以下主要部分：

```yaml
experiment:          # 实验配置
model:              # 模型配置
optimizer:          # 优化器配置
client:             # 客户端配置
data:               # 数据配置
aggregation:        # 聚合配置
evaluation:         # 评估配置
wandb:              # 日志配置
```

### 5.2 创建模型配置文件

在`configs/`目录下创建`your_model.yaml`：

```yaml
# configs/your_model.yaml

experiment:
  name: "your_model_federated_learning"
  rounds: 50
  seed: 42

client:
  num_clients: 10
  local_epochs: 2
  learning_rate: 0.01
  samples_per_client: null  # null表示使用默认均匀分布

model:
  type: "your_model"
  
  # 模型特有参数
  input_dim: 784
  hidden_dim: 128
  output_dim: 10

optimizer:
  learning_rate: 0.001
  weight_decay: 0.01
  betas: [0.9, 0.999]
  eps: 1e-8

data:
  batch_size: 32
  data_dir: "./data"

aggregation:
  type: "fedavg"

communication:
  type: "local"

evaluation:
  evaluate_every: 5

wandb:
  enabled: true
  project: "your-model-federated-learning"
  offline: false
  tags: 
    - "your_model"
    - "federated"

device: "auto"  # 'auto', 'cpu', 'cuda'等
```

### 5.3 配置参数说明

创建一个参数说明表：

| 参数类别 | 参数名 | 类型 | 默认值 | 说明 |
|----------|--------|------|--------|------|
| 模型 | `input_dim` | int | 784 | 输入维度 |
| 模型 | `hidden_dim` | int | 128 | 隐藏层维度 |
| 模型 | `output_dim` | int | 10 | 输出维度 |

---


## 总结

按照本指南的步骤，您可以成功地在联邦学习项目中添加任何新模型：

1. ✅ **实现模型类**：继承BaseModel，实现所有抽象方法
2. ✅ **更新工厂类**：在ModelFactory中添加模型创建逻辑
3. ✅ **更新导出**：在__init__.py中导出新模型
4. ✅ **创建配置**：编写YAML配置文件
5. ✅ **编写示例**：提供使用示例代码
6. ✅ **编写文档**：详细的使用指南
7. ✅ **测试验证**：全面的功能测试

记住：
- **始终继承BaseModel**
- **实现所有抽象方法**
- **提供合理的默认值**
- **充分测试您的实现**

现在您已经掌握了在联邦学习项目中添加新模型的完整流程！
