# 优化器配置系统使用指南

## 概述

本系统提供了一个精简且可复用的优化器配置方案，支持在联邦学习实验中轻松切换和配置不同的优化器。

## 特性

- ✅ **精简设计**: 最少的代码实现最大的功能
- ✅ **易于配置**: 通过YAML文件直接配置优化器
- ✅ **高度可复用**: 所有模型都可以使用相同的配置系统
- ✅ **向后兼容**: 兼容原有的学习率配置方式
- ✅ **类型安全**: 自动验证优化器类型和参数

## 支持的优化器

| 优化器 | 类型标识 | 主要参数 | 适用场景 |
|--------|----------|----------|----------|
| SGD | `sgd` | learning_rate, momentum, weight_decay | 经典优化器，稳定可靠 |
| Adam | `adam` | learning_rate, betas, eps, weight_decay | 自适应学习率，收敛快 |
| AdamW | `adamw` | learning_rate, betas, eps, weight_decay | Adam改进版，更好的权重衰减 |
| RMSprop | `rmsprop` | learning_rate, alpha, eps, momentum | 适合RNN和非平稳环境 |

## 配置方式

### 1. 在配置文件中设置

```yaml
# 优化器设置
optimizer:
  type: "adam"              # 优化器类型
  learning_rate: 0.001      # 学习率
  betas: [0.9, 0.999]      # Adam参数
  weight_decay: 0.0001      # 权重衰减
```

### 2. SGD配置示例

```yaml
optimizer:
  type: "sgd"
  learning_rate: 0.01
  momentum: 0.9
  weight_decay: 0.0001
```

### 3. AdamW配置示例

```yaml
optimizer:
  type: "adamw"
  learning_rate: 0.0005
  betas: [0.9, 0.999]
  weight_decay: 0.01       # AdamW通常使用更大的weight_decay
```

## 代码集成

### 模型中使用

```python
from utils.optimizer_factory import OptimizerFactory

class MyModel(BaseModel):
    def __init__(self, optimizer_config=None):
        super().__init__(optimizer_config)
        
        # 创建模型结构
        self.model = nn.Sequential(...)
        
        # 自动创建优化器
        self.create_optimizer(self.model.parameters())
```

### 手动创建优化器

```python
from utils.optimizer_factory import OptimizerFactory

# 配置字典
config = {
    "type": "adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0001
}

# 创建优化器
optimizer = OptimizerFactory.create_optimizer(model.parameters(), config)
```

## 实验运行

### 1. 使用预定义配置

```bash
# 使用SGD优化器
python run_experiment.py --config configs/default.yaml

# 使用Adam优化器  
python run_experiment.py --config configs/mnist_adam.yaml

# 使用AdamW优化器
python run_experiment.py --config configs/mnist_adamw.yaml
```

### 2. 命令行覆盖配置

```bash
# 动态修改学习率
python run_experiment.py --config configs/default.yaml --optimizer.learning_rate 0.001

# 切换优化器类型
python run_experiment.py --config configs/default.yaml --optimizer.type adam
```

### 3. 批量比较实验

```bash
# 运行优化器比较实验
python run_optimizer_comparison.py
```

## 默认配置

每种优化器都有预设的默认参数：

```python
# 获取默认配置
default_sgd = OptimizerFactory.get_default_config('sgd')
default_adam = OptimizerFactory.get_default_config('adam')
```

### SGD默认配置
```yaml
type: sgd
learning_rate: 0.01
momentum: 0.9
weight_decay: 0.0001
```

### Adam默认配置
```yaml
type: adam
learning_rate: 0.001
betas: [0.9, 0.999]
eps: 1e-8
weight_decay: 0.0001
```

## 扩展新优化器

添加新优化器非常简单：

1. 在 `OptimizerFactory.OPTIMIZERS` 中添加映射
2. 实现对应的参数处理函数
3. 添加默认配置

```python
# 添加新优化器
OPTIMIZERS = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
    'new_optimizer': optim.NewOptimizer,  # 新增
}

# 添加参数处理
@staticmethod
def _get_new_optimizer_params(config):
    params = {}
    # 处理特定参数
    return params
```

## 最佳实践

1. **选择合适的优化器**:
   - SGD: 经典选择，适合大多数场景
   - Adam: 快速收敛，适合复杂模型
   - AdamW: 大模型训练的首选

2. **学习率设置**:
   - SGD: 0.01-0.1
   - Adam/AdamW: 0.0001-0.001
   - 根据模型和数据集调整

3. **权重衰减**:
   - 防止过拟合的重要手段
   - SGD/Adam: 0.0001-0.001
   - AdamW: 0.01-0.1 (可以设置更大)

4. **联邦学习特殊考虑**:
   - 客户端数据不均匀时，较小的学习率更稳定
   - 本地训练轮次较多时，考虑学习率衰减

## 故障排除

### 常见问题

1. **优化器类型不支持**
   ```
   ValueError: 不支持的优化器类型: xxx
   ```
   解决: 检查拼写，确保使用支持的类型

2. **参数配置错误**
   ```
   TypeError: unexpected keyword argument
   ```
   解决: 检查参数名称和取值范围

3. **向后兼容性问题**
   - 系统自动处理旧的 `learning_rate` 配置
   - 如果没有优化器配置，会使用默认SGD

## 演示和测试

运行演示脚本查看所有功能：

```bash
# 查看优化器工厂演示
python examples/optimizer_demo.py

# 运行简单测试
python -c "from utils.optimizer_factory import OptimizerFactory; print(OptimizerFactory.get_supported_optimizers())"
```
