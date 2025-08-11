# 扩展指南

本指南将帮助您扩展联邦学习框架，添加新功能和自定义组件。

## 扩展概览

框架提供了多个扩展点，您可以根据需要添加新功能：

| 扩展类型 | 难度 | 用途 |
|---------|------|------|
| [新模型](new_models.md) | 简单 | 添加新的机器学习模型 |
| [自定义聚合算法](custom_aggregation.md) | 中等 | 实现新的参数聚合策略 |
| [网络通信](network_communication.md) | 中等 | 实现真实的网络通信 |
| [新数据源](data_sources.md) | 简单 | 支持新的数据格式和来源 |
| [隐私保护](privacy_protection.md) | 困难 | 添加差分隐私等保护机制 |
| [容错机制](fault_tolerance.md) | 困难 | 处理客户端故障和网络问题 |

## 扩展原则

### 1. 遵循抽象接口
所有扩展都应该继承相应的基类并实现所有抽象方法：

```python
# 好的扩展方式
class MyCustomModel(BaseModel):
    def get_parameters(self):
        # 实现获取参数逻辑
        pass
    
    def set_parameters(self, params):
        # 实现设置参数逻辑
        pass
    
    # 实现其他抽象方法...

# 避免的方式
class BadModel:  # 没有继承BaseModel
    def my_method(self):
        pass
```

### 2. 保持向后兼容
新功能应该不影响现有代码的运行：

```python
# 好的扩展方式
class EnhancedClient(FederatedClient):
    def __init__(self, client_id, model, data_loader=None, 
                 new_feature=None):  # 新参数有默认值
        super().__init__(client_id, model, data_loader)
        self.new_feature = new_feature

# 避免的方式
class BadClient(FederatedClient):
    def __init__(self, client_id, model, data_loader, 
                 required_new_param):  # 破坏了现有接口
        super().__init__(client_id, model, data_loader)
        self.required_new_param = required_new_param
```

### 3. 充分的文档和测试
每个扩展都应该包含：
- 详细的docstring文档
- 使用示例
- 单元测试
- 性能基准测试（如适用）

### 4. 模块化设计
新功能应该独立封装，避免与现有模块紧耦合：

```python
# 好的模块化设计
class PrivacyModule:
    """独立的隐私保护模块"""
    
    def add_noise(self, params, privacy_budget):
        # 隐私保护逻辑
        pass

class PrivacyPreservingClient(FederatedClient):
    def __init__(self, client_id, model, data_loader, privacy_module=None):
        super().__init__(client_id, model, data_loader)
        self.privacy_module = privacy_module or PrivacyModule()
```

## 快速开始扩展

### 1. 创建扩展文件夹
```bash
mkdir extensions/my_extension
touch extensions/my_extension/__init__.py
touch extensions/my_extension/my_component.py
```

### 2. 实现扩展组件
```python
# extensions/my_extension/my_component.py
from core.base import BaseClient

class MyExtendedClient(BaseClient):
    """我的扩展客户端"""
    
    def __init__(self, client_id, special_config):
        super().__init__(client_id)
        self.special_config = special_config
    
    def train(self, global_model_params):
        """实现训练逻辑"""
        pass
    
    def set_data(self, data):
        """实现数据设置逻辑"""
        pass
```

### 3. 编写测试
```python
# tests/test_my_extension.py
import unittest
from extensions.my_extension.my_component import MyExtendedClient

class TestMyExtendedClient(unittest.TestCase):
    def setUp(self):
        self.client = MyExtendedClient("test_client", {})
    
    def test_initialization(self):
        self.assertEqual(self.client.client_id, "test_client")
    
    def test_train(self):
        # 测试训练功能
        pass
```

### 4. 添加到主包
```python
# __init__.py
from extensions.my_extension.my_component import MyExtendedClient

# 现在可以直接导入使用
```

## 常见扩展模式

### 1. 装饰器模式
为现有组件添加新功能：

```python
class LoggingClientDecorator:
    """为客户端添加日志功能的装饰器"""
    
    def __init__(self, client, logger):
        self.client = client
        self.logger = logger
    
    def train(self, global_model_params):
        self.logger.info(f"Client {self.client.client_id} starting training")
        result = self.client.train(global_model_params)
        self.logger.info(f"Client {self.client.client_id} finished training")
        return result
    
    def __getattr__(self, name):
        # 代理其他方法调用
        return getattr(self.client, name)
```

### 2. 策略模式
支持可插拔的算法：

```python
class AggregationStrategy:
    """聚合策略接口"""
    
    def aggregate(self, client_updates, weights=None):
        raise NotImplementedError

class FedAvgStrategy(AggregationStrategy):
    def aggregate(self, client_updates, weights=None):
        # FedAvg实现
        pass

class FedProxStrategy(AggregationStrategy):
    def aggregate(self, client_updates, weights=None):
        # FedProx实现
        pass

class ConfigurableServer(FederatedServer):
    def __init__(self, model, strategy):
        super().__init__(model)
        self.strategy = strategy
    
    def aggregate(self, client_updates):
        return self.strategy.aggregate(client_updates, self.client_weights)
```

### 3. 观察者模式
监控训练过程：

```python
class TrainingObserver:
    """训练观察者接口"""
    
    def on_round_start(self, round_num):
        pass
    
    def on_round_end(self, round_num, metrics):
        pass
    
    def on_client_update(self, client_id, update):
        pass

class MetricsCollector(TrainingObserver):
    def __init__(self):
        self.metrics = []
    
    def on_round_end(self, round_num, metrics):
        self.metrics.append({
            'round': round_num,
            'metrics': metrics,
            'timestamp': time.time()
        })

class ObservableServer(FederatedServer):
    def __init__(self, model, aggregation_algorithm):
        super().__init__(model, aggregation_algorithm)
        self.observers = []
    
    def add_observer(self, observer):
        self.observers.append(observer)
    
    def notify_round_start(self, round_num):
        for observer in self.observers:
            observer.on_round_start(round_num)
    
    def notify_round_end(self, round_num, metrics):
        for observer in self.observers:
            observer.on_round_end(round_num, metrics)
```

## 性能考虑

### 1. 内存管理
```python
class MemoryEfficientClient(FederatedClient):
    """内存高效的客户端"""
    
    def train(self, global_model_params):
        # 使用生成器减少内存占用
        for batch in self._get_batch_generator():
            # 处理批次
            pass
        
        # 及时清理临时变量
        del temporary_variables
        
        return updated_params
    
    def _get_batch_generator(self):
        """使用生成器而不是列表"""
        for batch in self.data_loader:
            yield batch
```

### 2. 并行处理
```python
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

class ParallelServer(FederatedServer):
    """支持并行处理的服务器"""
    
    def __init__(self, model, aggregation_algorithm, num_workers=4):
        super().__init__(model, aggregation_algorithm)
        self.num_workers = num_workers
    
    def aggregate(self, client_updates):
        """并行聚合客户端更新"""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 并行处理客户端更新
            processed_updates = list(executor.map(
                self._process_update, client_updates
            ))
        
        return self._combine_updates(processed_updates)
    
    def _process_update(self, update):
        """处理单个客户端更新"""
        # 预处理逻辑
        return processed_update
    
    def _combine_updates(self, processed_updates):
        """合并处理后的更新"""
        # 合并逻辑
        pass
```

## 版本兼容性

### 1. 版本检查
```python
import sys
from packaging import version

def check_framework_version(required_version):
    """检查框架版本兼容性"""
    current_version = get_framework_version()
    if version.parse(current_version) < version.parse(required_version):
        raise RuntimeError(f"Required framework version {required_version}, "
                         f"but got {current_version}")

class VersionedExtension:
    """带版本检查的扩展"""
    
    def __init__(self):
        check_framework_version("0.1.0")
        # 扩展初始化逻辑
```

### 2. 特性检测
```python
def has_feature(feature_name):
    """检查框架是否支持特定特性"""
    try:
        import importlib
        module = importlib.import_module(f"features.{feature_name}")
        return True
    except ImportError:
        return False

class ConditionalExtension:
    """根据可用特性条件加载的扩展"""
    
    def __init__(self):
        if has_feature("advanced_crypto"):
            self.crypto_module = import_crypto_module()
        else:
            self.crypto_module = None
```

## 贡献扩展

如果您开发了有用的扩展，欢迎贡献给框架：

### 1. 准备贡献
- 确保代码遵循框架的编码规范
- 编写完整的测试用例
- 更新相关文档
- 添加使用示例

### 2. 提交流程
1. Fork 项目仓库
2. 创建功能分支
3. 实现扩展功能
4. 运行测试确保兼容性
5. 提交Pull Request

### 3. 代码审查标准
- 功能完整性
- 代码质量
- 测试覆盖率
- 文档完整性
- 性能影响评估

## 获取帮助

如果您在扩展过程中遇到问题：

1. 查看相关模块的文档
2. 参考现有的扩展示例
3. 查看测试用例了解用法
4. 提交Issue获取社区帮助

记住，良好的扩展应该遵循"开放-封闭原则"：对扩展开放，对修改封闭。
