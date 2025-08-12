# 客户端模块

客户端模块是联邦学习框架的核心组件之一，负责处理本地数据训练和与服务器的交互。

## 概述

客户端模块的主要职责：
- 维护本地模型状态
- 执行本地数据训练
- 与服务器交换模型参数
- 评估本地模型性能

## 核心类

### BaseClient (抽象基类)

所有客户端实现的基础接口。

```python
class BaseClient(ABC):
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.model = None
        self.data = None
    
    @abstractmethod
    def train(self, global_model_params: Dict[str, Any]) -> Dict[str, Any]:
        """训练本地模型"""
        pass
    
    @abstractmethod
    def set_data(self, data):
        """设置客户端数据"""
        pass
```

### FederatedClient (具体实现)

标准的联邦学习客户端实现。

#### 构造函数
```python
def __init__(self, 
             client_id: str, 
             model, 
             data_loader=None, 
             epochs=1, 
             learning_rate=0.01)
```

**参数说明:**
- `client_id`: 客户端唯一标识符
- `model`: 本地模型实例（继承自BaseModel）
- `data_loader`: 数据加载器
- `epochs`: 本地训练轮数
- `learning_rate`: 学习率

#### 主要方法

##### train()
执行本地训练的核心方法。

```python
def train(self, global_model_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    本地训练实现
    
    Args:
        global_model_params: 从服务器接收的全局模型参数
        
    Returns:
        训练后的本地模型参数
    """
```

**工作流程:**
1. 使用全局参数更新本地模型
2. 在本地数据上训练指定轮数
3. 返回更新后的模型参数

##### set_data()
设置客户端的训练数据。

```python
def set_data(self, data_loader):
    """设置数据加载器"""
```

##### evaluate()
评估本地模型性能。

```python
def evaluate(self, test_data, test_labels):
    """评估客户端模型"""
```

## 使用示例

### 基本使用
```python
from core.client import FederatedClient
from models.base import SimpleClassificationModel
from data.data_loader import FederatedDataLoader

# 创建模型
model = SimpleClassificationModel(
    input_dim=20, 
    num_classes=3, 
    learning_rate=0.01
)

# 创建数据加载器
data_loader = FederatedDataLoader(num_clients=5, batch_size=32)
client_dataloaders = data_loader.create_federated_data(
    data_type="classification",
    n_samples=1000,
    n_features=20,
    n_classes=3
)

# 创建客户端
client = FederatedClient(
    client_id="client_0",
    model=model,
    data_loader=client_dataloaders[0],
    epochs=3,
    learning_rate=0.01
)

# 执行训练
global_params = {}  # 从服务器获取
updated_params = client.train(global_params)
```

### 高级配置
```python
# 自定义训练配置
client = FederatedClient(
    client_id="advanced_client",
    model=custom_model,
    data_loader=custom_data_loader,
    epochs=5,              # 更多本地训练轮数
    learning_rate=0.001    # 较小的学习率
)

# 训练前设置特定数据
client.set_data(specialized_data_loader)

# 执行训练并评估
updated_params = client.train(global_params)
eval_results = client.evaluate(test_data, test_labels)
print(f"Client performance: {eval_results}")
```

## 扩展客户端

### 自定义客户端实现

```python
from core.base import BaseClient

class CustomClient(BaseClient):
    """自定义客户端实现"""
    
    def __init__(self, client_id: str, special_config):
        super().__init__(client_id)
        self.special_config = special_config
        # 自定义初始化逻辑
    
    def train(self, global_model_params):
        """自定义训练逻辑"""
        # 实现特殊的训练流程
        pass
    
    def set_data(self, data):
        """自定义数据设置"""
        # 实现特殊的数据处理
        pass
    
    def custom_method(self):
        """添加新功能"""
        pass
```

### 添加隐私保护

```python
class PrivacyPreservingClient(FederatedClient):
    """支持隐私保护的客户端"""
    
    def __init__(self, client_id, model, data_loader, 
                 privacy_budget=1.0, noise_multiplier=0.1):
        super().__init__(client_id, model, data_loader)
        self.privacy_budget = privacy_budget
        self.noise_multiplier = noise_multiplier
    
    def train(self, global_model_params):
        """带差分隐私的训练"""
        # 标准训练
        updated_params = super().train(global_model_params)
        
        # 添加噪声
        noisy_params = self._add_noise(updated_params)
        return noisy_params
    
    def _add_noise(self, params):
        """添加差分隐私噪声"""
        # 实现噪声添加逻辑
        pass
```

### 异步客户端

```python
import asyncio

class AsyncClient(BaseClient):
    """异步客户端实现"""
    
    async def train(self, global_model_params):
        """异步训练方法"""
        # 异步训练实现
        pass
    
    async def communicate_with_server(self, server_endpoint):
        """异步通信"""
        # 实现异步通信逻辑
        pass
```

## 配置选项

### 训练参数
- `epochs`: 本地训练轮数（1-10推荐）
- `learning_rate`: 学习率（0.001-0.1）
- `batch_size`: 通过数据加载器配置

### 数据配置
- 支持IID和Non-IID数据分布
- 可配置数据集大小和特征维度
- 支持自定义数据预处理

### 模型配置
- 支持分类和回归模型
- 可配置模型架构参数
- 支持模型检查点保存

## 性能优化

### 1. 批次大小优化
```python
# 根据内存情况调整批次大小
small_batch_client = FederatedClient(
    client_id="memory_limited",
    model=model,
    data_loader=DataLoader(dataset, batch_size=16),  # 小批次
    epochs=1
)
```

### 2. 本地训练轮数
```python
# 平衡通信成本和收敛速度
efficient_client = FederatedClient(
    client_id="efficient",
    model=model,
    data_loader=data_loader,
    epochs=3,  # 适中的本地训练轮数
    learning_rate=0.01
)
```

### 3. 模型参数压缩
```python
class CompressedClient(FederatedClient):
    """支持参数压缩的客户端"""
    
    def train(self, global_model_params):
        updated_params = super().train(global_model_params)
        # 压缩参数以减少通信开销
        compressed_params = self._compress_params(updated_params)
        return compressed_params
```

## 调试和监控

### 日志记录
```python
from utils.logger import default_logger

# 在客户端训练中添加日志
def train_with_logging(self, global_model_params):
    default_logger.info(f"Client {self.client_id} starting training")
    
    # 训练逻辑
    updated_params = self.train(global_model_params)
    
    default_logger.info(f"Client {self.client_id} completed training")
    return updated_params
```

### 性能监控
```python
import time

def monitored_train(self, global_model_params):
    start_time = time.time()
    
    updated_params = self.train(global_model_params)
    
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f}s")
    
    return updated_params
```

## 常见问题

### Q: 如何处理不同大小的客户端数据？
A: 框架会自动处理，聚合时可以考虑客户端数据量权重。

### Q: 客户端可以使用不同的模型架构吗？
A: 不建议，因为需要聚合相同结构的参数。如需要，可以实现自定义聚合算法。

### Q: 如何处理客户端掉线？
A: 当前版本使用本地通信，实际部署时需要实现容错机制。

### Q: 支持GPU训练吗？
A: 支持，模型会自动检测并使用可用的GPU设备。
