# 联邦学习框架

一个支持真实数据源的联邦学习框架，用于研究和实验环境。

## 主要特性

- **多种模型支持**: CNN模型、线性模型、MLP等，支持图像分类和结构化数据
- **真实数据支持**: 支持MNIST等
- **多种数据分布**: 支持IID和Non-IID数据分布，包括Dirichlet分布
- **灵活配置系统**: YAML配置文件 + 命令行参数覆盖
- **实验跟踪**: 集成WandB支持，便于实验管理和结果可视化
- **模块化设计**: 可扩展的客户端、服务器和聚合算法
- **简单易用**: 提供开箱即用的示例和工具

## 支持的数据源

### 1. CSV文件数据集
- 支持分类和回归任务
- 自动数据预处理和标准化
- 灵活的数据分割策略

### 2. 图像数据集  
- 支持文件夹结构的图像数据
- 自动图像预处理和变换
- 支持常见图像格式

### 3. 内置数据集
- MNIST手写数字识别
- CIFAR-10自然图像分类（计划中）
- Fashion-MNIST时尚用品分类（计划中）

## 项目结构

```
myWork/
├── __init__.py             # 框架初始化和便捷函数
├── core/                   # 核心模块
│   ├── __init__.py
│   ├── client.py          # 客户端实现
│   ├── server.py          # 服务器实现
│   └── base.py            # 基础抽象类
├── communication/          # 通信模块
│   ├── __init__.py
│   └── local.py           # 本地通信实现
├── models/                 # 模型模块
│   ├── __init__.py
│   ├── base.py            # 基础模型类
│   └── cnn.py             # CNN模型实现
├── data/                   # 数据处理模块
│   ├── __init__.py
│   └── data_loader.py     # 数据加载器
├── aggregation/            # 聚合算法模块
│   ├── __init__.py
│   └── federated_avg.py   # FedAvg聚合算法
├── utils/                  # 工具模块
│   ├── __init__.py
│   ├── logger.py          # 日志工具
│   ├── config_manager.py   # 配置管理器
│   ├── experiment_runner.py # 实验执行器
│   ├── model_factory.py    # 模型工厂
│   ├── results_handler.py  # 结果处理器
│   └── wandb_logger.py     # WandB日志
├── configs/                # 配置文件
│   ├── default.yaml       # 默认配置
│   ├── mnist.yaml         # MNIST配置
│   ├── mnist_wandb.yaml   # MNIST WandB配置
│   └── non_iid.yaml       # Non-IID配置
├── examples/               # 示例代码
│   ├── config_manager_example.py
│   ├── config_usage_example.py
│   ├── data_transform_example.py
│   ├── model_extension_example.py
│   ├── run_experiments.py
│   └── simple_federated_mnist_experiment.py
├── docs/                   # 文档
│   ├── README.md
│   ├── quick_start.md     # 快速开始
│   ├── data_guide.md      # 数据使用指南
│   ├── architecture.md    # 架构文档
│   ├── experiment_guide.md # 实验指南
│   ├── data_transform_best_practices.md
│   ├── refactoring_guide.md
│   └── wandb_guide.md     # WandB使用指南
├── run_experiment.py       # 主实验入口
├── quick_start.py          # 快速开始示例
├── experiment.py           # 实验模块
├── validate_config.py      # 配置验证
├── test_wandb.py          # WandB测试
├── test_wandb_timeout.py  # WandB超时处理测试
├── demo_wandb_handling.py # WandB超时处理演示
├── check_environment.py    # 环境验证脚本
├── requirements.txt        # 依赖
└── setup.py               # 安装脚本
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```


### 2. 运行示例

#### 基础MNIST联邦学习实验（使用CNN模型）

```bash
# 使用默认CNN模型和MNIST数据集进行联邦学习
python run_experiment.py --config configs/mnist.yaml

# 或使用命令行参数自定义配置
python run_experiment.py --config configs/mnist.yaml --rounds 10 --num-clients 5 --local-epochs 3
```

#### 快速测试实验

```bash
# 运行快速测试（少轮次、少客户端）
python run_experiment.py --config configs/mnist.yaml --rounds 2 --num-clients 2 --selected-clients 1
```

#### 使用独立示例脚本

```bash
# 运行简单的CNN联邦学习MNIST实验
python examples/simple_federated_mnist_experiment.py

# 查看所有实验示例
python examples/run_experiments.py
```

#### Non-IID数据分布实验

```bash
# 使用Non-IID数据分布进行实验
python run_experiment.py --config configs/non_iid.yaml
```

### 3. 使用WandB进行实验跟踪（支持超时处理）

```bash
# 使用WandB记录实验结果，支持自动超时处理和离线模式
python run_experiment.py --config configs/mnist.yaml

# 测试WandB超时处理功能
python demo_wandb_handling.py --timeout 10 --rounds 3

# 手动设置超时时间
python run_experiment.py --config configs/mnist.yaml --wandb.init_timeout=120
```

**WandB 特色功能**：
- 🚀 自动超时处理：初始化超时时自动切换离线模式
- 📶 智能降级：网络问题时自动使用离线模式
- 🔄 自动同步：实验结束后自动尝试同步离线数据
- 💪 容错能力：WandB 问题不会中断实验执行

## CNN模型 + MNIST数据集联邦学习示例

### 推荐的启动命令

```bash
# 标准CNN联邦学习实验（推荐配置）
python run_experiment.py --config configs/mnist.yaml

# 详细参数说明：
# - 使用CNN模型进行图像分类
# - MNIST手写数字数据集
# - 10个客户端，每轮选择5个客户端参与训练
# - 20轮联邦学习
# - 每个客户端本地训练3个epoch
# - 使用Non-IID数据分布（Dirichlet分布，α=0.3）
# - 学习率：0.01
```

### 自定义参数示例

```bash
# 调整轮数和客户端数量
python run_experiment.py --config configs/mnist.yaml --rounds 15 --num-clients 8 --selected-clients 4

# 调整学习参数
python run_experiment.py --config configs/mnist.yaml --learning-rate 0.001 --local-epochs 5

# 使用IID数据分布
python run_experiment.py --config configs/mnist.yaml --iid

# 快速验证实验
python run_experiment.py --config configs/mnist.yaml --rounds 3 --num-clients 3 --selected-clients 2
```

### 使用独立示例脚本

```bash
# 运行简化的CNN联邦学习示例（固定参数）
python examples/simple_federated_mnist_experiment.py

# 该示例包含：
# - 3个客户端
# - 5轮联邦学习
# - CNN模型
# - 本地训练2个epoch
```

## 配置文件说明

项目提供多个预配置的YAML文件，位于`configs/`目录：

- **`mnist.yaml`**: 标准MNIST CNN联邦学习配置
  - 10个客户端，每轮选择5个
  - 20轮联邦学习，每个客户端本地训练3个epoch
  - CNN模型，学习率0.01
  - Non-IID数据分布（Dirichlet α=0.3）

- **`mnist_wandb.yaml`**: 带WandB日志记录的MNIST配置
- **`non_iid.yaml`**: 专门的Non-IID数据分布配置
- **`default.yaml`**: 默认配置模板

### 命令行参数覆盖

支持的命令行参数：
- `--rounds`: 联邦学习轮数
- `--num-clients`: 客户端总数
- `--selected-clients`: 每轮选择的客户端数
- `--local-epochs`: 客户端本地训练轮数
- `--learning-rate`: 学习率
- `--batch-size`: 批次大小
- `--iid`: 使用IID数据分布

## 总结：CNN + MNIST联邦学习启动命令

以下是使用CNN模型和MNIST数据集进行联邦学习的完整命令示例：

### 🎯 推荐启动命令

```bash
# 标准配置 - 推荐新手使用
python run_experiment.py --config configs/mnist.yaml

# 这个命令将启动：
# ✅ CNN模型（专为MNIST设计的卷积神经网络）
# ✅ MNIST手写数字数据集（自动下载）
# ✅ 10个联邦客户端，每轮随机选择5个参与训练
# ✅ 20轮联邦学习训练
# ✅ Non-IID数据分布（更贴近真实场景）
# ✅ 每2轮进行一次全局模型评估
```

### 🚀 快速验证命令

```bash
# 快速测试 - 用于验证环境配置
python run_experiment.py --config configs/mnist.yaml --rounds 3 --num-clients 3
```

### ⚙️ 高级自定义命令

```bash
# 完全自定义配置
python run_experiment.py --config configs/mnist.yaml \
    --rounds 25 \
    --num-clients 20 \
    --selected-clients 8 \
    --local-epochs 5 \
    --learning-rate 0.001 \
    --batch-size 128
```

### 📊 带实验跟踪的命令

```bash
# 使用WandB记录实验过程（需要WandB账户）
python run_experiment.py --config configs/mnist_wandb.yaml
```

## 设计原则

- **真实性**: 使用真实数据集，避免合成数据的局限性
- **模块化**: 各组件可独立替换和扩展
- **简洁性**: 专注核心功能，易于理解和修改
- **灵活性**: 支持多种数据类型和分布模式

## 文档

- [快速开始](docs/quick_start.md) - 入门教程和安装指南
- [数据使用指南](docs/data_guide.md) - 详细的数据加载和使用说明
- [实验指南](docs/experiment_guide.md) - 实验配置和运行指南
- [架构文档](docs/architecture.md) - 系统架构说明
- [WandB使用指南](docs/wandb_guide.md) - 实验跟踪和可视化
- [数据变换最佳实践](docs/data_transform_best_practices.md)
- [重构指南](docs/refactoring_guide.md)

## 支持的功能

### 数据类型
- ✅ CSV结构化数据
- ✅ 图像数据（文件夹结构）
- ✅ 内置数据集（MNIST、CIFAR-10等）
- ✅ 分类和回归任务

### 数据分布
- ✅ IID（独立同分布）
- ✅ Non-IID（非独立同分布）
- ✅ 自定义分割策略

### 模型支持
- ✅ CNN模型（用于图像分类）
- ✅ 简单线性模型
- ✅ 多层感知机
- ✅ 自定义模型扩展

### 聚合算法
- ✅ FedAvg（联邦平均）
- 🔄 FedProx（计划中）
- 🔄 自定义聚合（计划中）

### 实验管理
- ✅ 配置文件管理（YAML格式）
- ✅ 命令行参数覆盖
- ✅ 实验结果记录
- ✅ WandB集成支持
- ✅ 实验配置验证


- **模块化**：每个模块职责单一，便于扩展
- **可扩展**：通过继承基类可以轻松添加新功能
- **最小化**：只包含核心功能，避免冗余
- **解耦**：模块间通过接口交互，降低耦合度
