# 服务器模块

服务器模块是联邦学习的协调中心，负责管理全局模型、聚合客户端更新并协调整个训练过程。

## 概述

服务器模块的主要职责：
- 维护全局模型状态
- 协调客户端训练过程
- 聚合客户端模型更新
- 评估全局模型性能
- 决定训练终止条件

## 核心类

### BaseServer (抽象基类)

所有服务器实现的基础接口。

```python
class BaseServer(ABC):
    def __init__(self):
        self.global_model = None
        self.clients = []
    
    @abstractmethod
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚合客户端更新"""
        pass
    
    @abstractmethod
    def initialize_model(self):
        """初始化全局模型"""
        pass
```

### FederatedServer (具体实现)

标准的联邦学习服务器实现。

#### 构造函数
```python
def __init__(self, model, aggregation_algorithm=None)
```

**参数说明:**
- `model`: 全局模型实例（继承自BaseModel）
- `aggregation_algorithm`: 聚合算法实例（如FederatedAveraging）

#### 主要方法

##### aggregate()
聚合客户端模型更新的核心方法。

```python
def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    聚合客户端更新
    
    Args:
        client_updates: 客户端模型参数更新列表
        
    Returns:
        聚合后的全局模型参数
    """
```

**工作流程:**
1. 使用聚合算法处理客户端更新
2. 更新全局模型参数
3. 增加训练轮次计数
4. 返回新的全局模型参数

##### send_global_model()
发送全局模型参数给客户端。

```python
def send_global_model(self) -> Dict[str, Any]:
    """发送全局模型参数给客户端"""
```

##### set_client_weights()
设置客户端权重（用于加权聚合）。

```python
def set_client_weights(self, client_weights: Dict[str, float]):
    """设置客户端权重"""
```

##### evaluate_global_model()
评估全局模型性能。

```python
def evaluate_global_model(self, test_data, test_labels):
    """评估全局模型"""
```

## 使用示例

### 基本使用
```python
from core.server import FederatedServer
from models.base import SimpleClassificationModel
from aggregation.federated_avg import FederatedAveraging

# 创建全局模型
global_model = SimpleClassificationModel(
    input_dim=20,
    num_classes=3,
    learning_rate=0.01
)

# 创建聚合算法
aggregation_algorithm = FederatedAveraging()

# 创建服务器
server = FederatedServer(global_model, aggregation_algorithm)

# 获取初始参数
global_params = server.send_global_model()

# 聚合客户端更新
client_updates = [client1_params, client2_params, client3_params]
new_global_params = server.aggregate(client_updates)

# 评估全局模型
test_results = server.evaluate_global_model(test_data, test_labels)
```

### 加权聚合
```python
# 设置客户端权重（基于数据量）
client_weights = {
    "client_0": 0.3,  # 30%的数据
    "client_1": 0.5,  # 50%的数据
    "client_2": 0.2   # 20%的数据
}
server.set_client_weights(client_weights)

# 执行加权聚合
aggregated_params = server.aggregate(client_updates)
```

### 完整训练循环
```python
from utils.logger import default_logger

def run_federated_training(server, clients, communication, 
                          test_data, test_labels, num_rounds=10):
    """完整的联邦学习训练循环"""
    
    for round_num in range(1, num_rounds + 1):
        default_logger.log_round_start(round_num)
        
        # 1. 广播全局模型参数
        global_params = server.send_global_model()
        communication.broadcast_to_clients(global_params)
        
        # 2. 客户端训练
        client_updates = []
        for client in clients:
            messages = communication.receive_from_server(client.client_id)
            latest_params = messages[-1] if messages else global_params
            updated_params = client.train(latest_params)
            client_updates.append(updated_params)
            communication.send_to_server(client.client_id, updated_params)
        
        # 3. 服务器聚合
        server.aggregate(client_updates)
        
        # 4. 评估和记录
        eval_results = server.evaluate_global_model(test_data, test_labels)
        default_logger.log_round_end(round_num, eval_results)
        
        # 5. 清理通信缓冲区
        communication.clear_buffers()
    
    return server.send_global_model()
```

## 扩展服务器

### 自定义服务器实现

```python
from core.base import BaseServer

class CustomServer(BaseServer):
    """自定义服务器实现"""
    
    def __init__(self, model, custom_config):
        super().__init__()
        self.global_model = model
        self.custom_config = custom_config
    
    def aggregate(self, client_updates):
        """自定义聚合逻辑"""
        # 实现特殊的聚合策略
        pass
    
    def initialize_model(self):
        """自定义模型初始化"""
        pass
    
    def custom_selection(self, all_clients):
        """客户端选择策略"""
        # 实现客户端选择逻辑
        pass
```

### 支持客户端选择的服务器

```python
import random

class SelectiveServer(FederatedServer):
    """支持客户端选择的服务器"""
    
    def __init__(self, model, aggregation_algorithm, 
                 selection_ratio=0.5, selection_strategy="random"):
        super().__init__(model, aggregation_algorithm)
        self.selection_ratio = selection_ratio
        self.selection_strategy = selection_strategy
        self.all_clients = []
    
    def register_clients(self, clients):
        """注册所有客户端"""
        self.all_clients = clients
    
    def select_clients(self):
        """选择参与训练的客户端"""
        num_selected = int(len(self.all_clients) * self.selection_ratio)
        
        if self.selection_strategy == "random":
            return random.sample(self.all_clients, num_selected)
        elif self.selection_strategy == "performance":
            # 基于性能选择客户端
            return self._select_by_performance(num_selected)
        else:
            return self.all_clients[:num_selected]
    
    def _select_by_performance(self, num_selected):
        """基于性能选择客户端"""
        # 实现基于历史性能的选择策略
        pass
```

### 自适应聚合服务器

```python
class AdaptiveServer(FederatedServer):
    """自适应聚合服务器"""
    
    def __init__(self, model, aggregation_algorithm):
        super().__init__(model, aggregation_algorithm)
        self.client_performance_history = {}
        self.adaptive_weights = {}
    
    def aggregate(self, client_updates):
        """自适应聚合"""
        # 根据客户端历史性能调整权重
        self._update_adaptive_weights(client_updates)
        
        # 使用自适应权重进行聚合
        self.set_client_weights(self.adaptive_weights)
        return super().aggregate(client_updates)
    
    def _update_adaptive_weights(self, client_updates):
        """更新自适应权重"""
        # 实现自适应权重计算逻辑
        pass
```

## 高级功能

### 1. 模型版本管理

```python
class VersionedServer(FederatedServer):
    """支持模型版本管理的服务器"""
    
    def __init__(self, model, aggregation_algorithm):
        super().__init__(model, aggregation_algorithm)
        self.model_history = []
        self.best_model = None
        self.best_performance = float('-inf')
    
    def aggregate(self, client_updates):
        """聚合并保存模型版本"""
        new_params = super().aggregate(client_updates)
        
        # 保存模型快照
        self.model_history.append({
            'round': self.round_num,
            'params': copy.deepcopy(new_params),
            'timestamp': time.time()
        })
        
        return new_params
    
    def evaluate_and_save_best(self, test_data, test_labels):
        """评估并保存最佳模型"""
        current_performance = self.evaluate_global_model(test_data, test_labels)
        
        if current_performance['accuracy'] > self.best_performance:
            self.best_performance = current_performance['accuracy']
            self.best_model = copy.deepcopy(self.send_global_model())
        
        return current_performance
```

### 2. 早停机制

```python
class EarlyStoppingServer(FederatedServer):
    """支持早停的服务器"""
    
    def __init__(self, model, aggregation_algorithm, 
                 patience=5, min_delta=0.001):
        super().__init__(model, aggregation_algorithm)
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.should_stop = False
    
    def check_early_stopping(self, current_loss):
        """检查是否需要早停"""
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.should_stop = True
            
        return self.should_stop
```

### 3. 联邦学习调度器

```python
class FederatedScheduler:
    """联邦学习调度器"""
    
    def __init__(self, server, clients, communication):
        self.server = server
        self.clients = clients
        self.communication = communication
        self.round_num = 0
        self.training_history = []
    
    def run_training(self, num_rounds, test_data=None, test_labels=None):
        """运行联邦学习训练"""
        for round_num in range(1, num_rounds + 1):
            self.round_num = round_num
            
            # 执行一轮训练
            round_results = self._execute_round()
            
            # 记录训练历史
            self.training_history.append(round_results)
            
            # 评估和记录
            if test_data is not None:
                eval_results = self.server.evaluate_global_model(
                    test_data, test_labels
                )
                round_results['evaluation'] = eval_results
            
            # 检查早停条件
            if hasattr(self.server, 'should_stop') and self.server.should_stop:
                break
        
        return self.training_history
    
    def _execute_round(self):
        """执行单轮训练"""
        # 实现单轮训练逻辑
        pass
```

## 性能优化

### 1. 异步聚合
```python
import asyncio

class AsyncServer(FederatedServer):
    """异步服务器实现"""
    
    async def async_aggregate(self, client_updates):
        """异步聚合"""
        # 实现异步聚合逻辑
        pass
    
    async def wait_for_clients(self, timeout=30):
        """等待客户端更新（带超时）"""
        # 实现客户端等待逻辑
        pass
```

### 2. 增量聚合
```python
class IncrementalServer(FederatedServer):
    """增量聚合服务器"""
    
    def __init__(self, model, aggregation_algorithm):
        super().__init__(model, aggregation_algorithm)
        self.received_updates = []
    
    def add_client_update(self, client_id, update):
        """添加客户端更新"""
        self.received_updates.append((client_id, update))
        
        # 如果收到足够的更新，立即聚合
        if len(self.received_updates) >= self.min_clients_for_aggregation:
            self._perform_incremental_aggregation()
    
    def _perform_incremental_aggregation(self):
        """执行增量聚合"""
        pass
```

## 监控和调试

### 训练监控
```python
class MonitoredServer(FederatedServer):
    """支持监控的服务器"""
    
    def __init__(self, model, aggregation_algorithm):
        super().__init__(model, aggregation_algorithm)
        self.metrics_history = []
        self.convergence_threshold = 0.001
    
    def aggregate(self, client_updates):
        """聚合并监控收敛"""
        old_params = copy.deepcopy(self.send_global_model())
        new_params = super().aggregate(client_updates)
        
        # 计算参数变化
        param_change = self._calculate_param_change(old_params, new_params)
        self.metrics_history.append({
            'round': self.round_num,
            'param_change': param_change,
            'num_clients': len(client_updates)
        })
        
        return new_params
    
    def _calculate_param_change(self, old_params, new_params):
        """计算参数变化幅度"""
        # 实现参数变化计算
        pass
    
    def is_converged(self):
        """检查是否收敛"""
        if len(self.metrics_history) < 3:
            return False
        
        recent_changes = [m['param_change'] for m in self.metrics_history[-3:]]
        return all(change < self.convergence_threshold for change in recent_changes)
```

## 配置选项

### 聚合策略
- `FederatedAveraging`: 标准联邦平均
- 自定义聚合算法（通过继承实现）

### 客户端管理
- 客户端选择比例
- 最小参与客户端数量
- 客户端权重策略

### 训练控制
- 最大训练轮数
- 早停条件
- 收敛阈值

## 常见问题

### Q: 如何处理客户端更新延迟？
A: 可以实现异步聚合或设置超时机制。

### Q: 服务器可以使用不同的聚合算法吗？
A: 可以，通过传入不同的聚合算法实例即可。

### Q: 如何实现模型的持久化？
A: 可以在聚合后保存模型参数到文件或数据库。

### Q: 支持多GPU训练吗？
A: 当前版本主要针对单机模拟，多GPU支持需要额外实现。
