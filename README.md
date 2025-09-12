# 联邦学习框架

一个支持多模态模型的现代化联邦学习框架，专为研究和实验环境设计。

## 主要特性

- **多模态模型支持**: 支持CLIP多模态模型、CNN模型、线性模型等，涵盖图像分类和多模态任务
- **真实数据支持**: 支持MNIST、CIFAR-10等经典数据集
- **多种数据分布**: 支持IID和Non-IID数据分布，包括Dirichlet分布
- **灵活配置系统**: YAML配置文件 + 命令行参数覆盖，支持模型特定配置
- **实验跟踪**: 集成WandB支持，提供智能超时处理和离线模式
- **模块化设计**: 可扩展的客户端、服务器和聚合算法
- **多模态能力**: 支持文本-图像联合训练的CLIP模型

## 支持的数据源

### 1. 经典数据集（已支持）
- **MNIST**: 手写数字识别，支持Non-IID分布
- **CIFAR-10**: 自然图像分类，支持多模态联邦学习

### 2. 多模态数据（CLIP支持）
- 图像-文本对比学习数据
- 支持预训练CLIP模型的微调
- 自动图像预处理和文本编码

### 3. 数据分布策略
- **IID分布**: 数据在客户端间均匀分布
- **Non-IID分布**: 基于Dirichlet分布的不平衡数据分配
- **多数据集混合**: 支持客户端使用多种数据集

## 项目结构

```
myWork/
├── main.py                 # 主实验入口
├── README.md              # 项目说明文档
├── requirements.txt       # 项目依赖
├── core/                  # 核心模块
│   ├── __init__.py
│   ├── client.py         # 客户端实现
│   ├── server.py         # 服务器实现
│   └── base.py           # 基础抽象类
├── communication/         # 通信模块
│   ├── __init__.py
│   └── local.py          # 本地通信实现
├── models/                # 模型模块
│   ├── __init__.py
│   ├── base.py           # 基础模型类
│   ├── cnn.py            # CNN模型实现
│   └── clip.py           # CLIP多模态模型（新增）
├── data/                  # 数据处理模块
│   ├── __init__.py
│   ├── data_loader.py    # 数据加载器
│   ├── middleware.py     # 数据中间件
│   └── datasets/         # 数据集实现
│       ├── mnist.py      # MNIST数据集
│       └── cifar10.py    # CIFAR-10数据集
├── aggregation/           # 聚合算法模块
│   ├── __init__.py
│   └── federated_avg.py  # FedAvg聚合算法
├── utils/                 # 工具模块
│   ├── __init__.py
│   ├── config_manager.py  # 配置管理器
│   ├── experiment_runner.py # 实验执行器
│   ├── model_factory.py   # 模型工厂
│   ├── optimizer_factory.py # 优化器工厂
│   ├── results_handler.py # 结果处理器
│   ├── wandb_logger.py    # WandB日志
│   ├── device_manager.py  # 设备管理器
│   └── evaluation_manager.py # 评估管理器
├── configs/               # 配置文件目录
│   ├── default.yaml      # 默认配置模板
│   ├── template.yaml     # 配置模板文件
│   ├── mnist.yaml        # MNIST实验配置
│   ├── simple_mnist.yaml # 简化MNIST配置
│   ├── clip.yaml         # CLIP模型配置（新增）
│   ├── clip_multimodal.yaml # CLIP多模态配置（新增）
│   ├── multi_dataset.yaml # 多数据集配置
│   └── validate_configs.py # 配置验证工具
├── test/                  # 测试目录
├── tests/                 # 单元测试
└── wandb/                 # WandB实验记录
```

## 快速开始

### 1. 环境配置

```bash
# 安装项目依赖
pip install -r requirements.txt
```

**核心依赖说明**：
- `torch>=1.9.0` + `torchvision>=0.10.0`: PyTorch深度学习框架
- `transformers>=4.30.0`: Hugging Face变换器库（CLIP模型支持）
- `wandb>=0.15.0`: 实验跟踪和可视化
- `PyYAML>=6.0`: 配置文件解析

### 2. 基础实验运行

#### 📝 MNIST CNN联邦学习（推荐入门）

```bash
# 使用CNN模型进行MNIST联邦学习
python main.py --config configs/mnist.yaml

# 快速验证实验（3轮训练）
python main.py --config configs/simple_mnist.yaml
```

#### 🌟 CLIP多模态联邦学习（新特性）

```bash
# 使用CLIP模型进行多模态联邦学习
python main.py --config configs/clip.yaml

# CLIP多数据集联邦学习
python main.py --config configs/clip_multimodal.yaml
```

#### ⚙️ 自定义参数运行

```bash
# 调整训练参数
python main.py --config configs/mnist.yaml --rounds 15 --num-clients 8 --local-epochs 3

# 使用IID数据分布
python main.py --config configs/mnist.yaml --iid

# 启用WandB实验跟踪
python main.py --config configs/mnist.yaml --wandb.enabled=true
```

### 3. 实验配置说明

#### 🔧 可用配置文件

| 配置文件 | 用途 | 模型类型 | 数据集 | 特点 |
|---------|-----|---------|-------|------|
| `mnist.yaml` | 标准MNIST实验 | CNN | MNIST | 基础联邦学习实验 |
| `simple_mnist.yaml` | 快速验证 | CNN | MNIST | 少轮次快速测试 |
| `clip.yaml` | CLIP单模态 | CLIP | MNIST/CIFAR-10 | 多模态模型图像分类 |
| `clip_multimodal.yaml` | CLIP多模态 | CLIP | 多数据集 | 真正的多模态联邦学习 |
| `multi_dataset.yaml` | 多数据集混合 | CNN | MNIST+CIFAR-10 | 异构数据联邦学习 |
| `default.yaml` | 配置模板 | 可配置 | 可配置 | 自定义实验基础 |

#### ⚡ WandB实验跟踪（智能超时处理）

```bash
# 启用WandB实验记录
python main.py --config configs/clip.yaml --wandb.enabled=true

# WandB特色功能：
# ✅ 自动超时处理：网络问题时自动切换离线模式
# ✅ 智能降级：初始化失败时不中断实验
# ✅ 自动同步：实验结束后尝试同步离线数据
```

## CLIP多模态联邦学习（新特性）

### 🌟 CLIP模型优势

- **预训练优势**: 基于OpenAI CLIP预训练模型，具备强大的视觉-语言理解能力
- **多模态支持**: 同时处理图像和文本信息，适合复杂的联邦学习场景
- **迁移学习**: 在联邦环境中微调预训练模型，提升收敛速度
- **模块化设计**: 支持独立的图像编码器、文本编码器和分类头

### 🚀 CLIP联邦学习启动命令

```bash
# 标准CLIP联邦学习（推荐）
python main.py --config configs/clip.yaml

# CLIP多模态实验（文本+图像）
python main.py --config configs/clip_multimodal.yaml

# 自定义CLIP参数
python main.py --config configs/clip.yaml \
    --rounds 25 \
    --num-clients 6 \
    --learning-rate 1e-5 \
    --model.freeze_encoder=false
```

### 📋 CLIP配置详解

```yaml
model:
  type: "clip"                             # 模型类型
  model_name: "openai/clip-vit-base-patch32"  # 预训练模型
  cache_dir: "/home/zzm/checkpoint"        # 模型缓存目录
  freeze_encoder: false                    # 是否冻结编码器
  normalize_features: true                 # 特征归一化

optimizer:
  type: "adamw"
  learning_rate: 5e-5                      # CLIP推荐学习率
  weight_decay: 0.1                        # 较大权重衰减
  betas: [0.9, 0.98]                      # CLIP优化参数
```

## 配置文件详解

### 📁 配置文件结构说明

每个YAML配置文件包含以下主要部分：

```yaml
# 实验基础设置
experiment:
  name: "实验名称"
  rounds: 10              # 联邦学习轮数
  seed: 42               # 随机种子

# 客户端配置
client:
  num_clients: 5          # 客户端总数
  local_epochs: 3         # 本地训练轮数
  learning_rate: 0.001    # 学习率
  client_datasets:        # 客户端数据分配
    client_0:
      - mnist

# 模型配置
model:
  type: "cnn"            # 模型类型：cnn/clip
  learning_rate: 0.001   # 模型学习率

# 数据配置
data:
  data_dir: "/path/to/data"
  batch_size: 64
  datasets:
    mnist:
      train: true

# WandB配置
wandb:
  enabled: false         # 是否启用实验跟踪
  project: "federated-learning"
```

### 🔧 命令行参数覆盖

支持通过命令行参数覆盖配置文件中的任何设置：

```bash
# 基础参数覆盖
python main.py --config configs/mnist.yaml \
    --rounds 20 \
    --num-clients 10 \
    --local-epochs 5 \
    --learning-rate 0.01

# 嵌套参数覆盖（使用点号）
python main.py --config configs/clip.yaml \
    --model.freeze_encoder=true \
    --wandb.enabled=true \
    --data.batch_size=32
```

## 实验示例汇总

### 🎯 推荐实验运行顺序

#### 1. 入门验证实验
```bash
# 快速验证环境配置（约2-3分钟）
python main.py --config configs/simple_mnist.yaml
```

#### 2. 标准CNN联邦学习
```bash
# 经典MNIST CNN联邦学习（约10-15分钟）
python main.py --config configs/mnist.yaml
```

#### 3. CLIP多模态实验
```bash
# CLIP模型联邦学习（需GPU，约20-30分钟）
python main.py --config configs/clip.yaml

# CLIP多数据集联邦学习
python main.py --config configs/clip_multimodal.yaml
```

#### 4. 高级自定义实验
```bash
# 多数据集异构联邦学习
python main.py --config configs/multi_dataset.yaml

# 自定义大规模实验
python main.py --config configs/default.yaml \
    --rounds 50 \
    --num-clients 20 \
    --wandb.enabled=true
```

### 📊 实验结果说明

每次实验结束后，会显示：
- **训练损失曲线**: 每轮全局模型的训练损失
- **测试准确率**: 各数据集的测试准确率
- **客户端统计**: 每个客户端的数据分布情况
- **模型参数**: 总参数量和可训练参数数量

## 技术特性

### 🏗️ 架构设计原则

- **模块化**: 各组件独立可替换，支持自定义扩展
- **多模态**: 支持传统CNN和现代CLIP多模态模型
- **可配置**: 通过YAML配置文件灵活控制实验参数
- **容错性**: WandB智能超时处理，网络问题不影响实验
- **扩展性**: 基于工厂模式的模型和优化器创建

### 🔧 支持的功能矩阵

| 功能类别 | 支持状态 | 说明 |
|---------|---------|------|
| **模型类型** | | |
| ✅ CNN模型 | 完全支持 | 卷积神经网络，适用于图像分类 |
| ✅ CLIP模型 | 完全支持 | 多模态预训练模型，支持视觉-语言任务 |
| ✅ 线性模型 | 完全支持 | 简单全连接网络 |
| **数据类型** | | |
| ✅ MNIST | 完全支持 | 手写数字识别 |
| ✅ CIFAR-10 | 完全支持 | 自然图像分类 |
| 🔄 多模态数据 | 开发中 | 图像-文本对比学习数据 |
| **数据分布** | | |
| ✅ IID分布 | 完全支持 | 数据均匀分布 |
| ✅ Non-IID分布 | 完全支持 | Dirichlet分布 |
| ✅ 多数据集混合 | 完全支持 | 客户端使用不同数据集 |
| **聚合算法** | | |
| ✅ FedAvg | 完全支持 | 联邦平均算法 |
| 🔄 FedProx | 计划中 | 针对异构数据的算法 |
| **实验管理** | | |
| ✅ YAML配置 | 完全支持 | 灵活的配置文件系统 |
| ✅ 命令行覆盖 | 完全支持 | 动态参数调整 |
| ✅ WandB集成 | 完全支持 | 智能超时处理的实验跟踪 |

## 开发指南

### 🔧 添加新模型

1. 在`models/`目录下创建新模型文件
2. 继承`BaseModel`类并实现必要方法
3. 在`utils/model_factory.py`中注册新模型
4. 创建对应的配置文件

```python
# 示例：添加新模型
from core.base import BaseModel

class MyCustomModel(BaseModel):
    def __init__(self, optimizer_config=None):
        super().__init__(optimizer_config)
        # 实现模型初始化
        
    def forward(self, x):
        # 实现前向传播
        pass
```

### 📊 添加新数据集

1. 在`data/datasets/`目录下实现数据集类
2. 在`data/data_loader.py`中添加数据集支持
3. 更新配置文件模板

### ⚙️ 配置验证

使用内置的配置验证工具：

```bash
# 验证所有配置文件
python configs/validate_configs.py

# 验证特定配置文件
python configs/validate_configs.py configs/clip.yaml
```

### 🧪 测试和调试

```bash
# 运行单元测试
python -m pytest tests/

# 快速功能测试
python main.py --config configs/simple_mnist.yaml --rounds 1
```

## 常见问题解答

### Q: CLIP模型运行时出现内存不足怎么办？
A: 尝试以下解决方案：
- 减小批次大小：`--data.batch_size=8`
- 减少客户端数量：`--num-clients=2`
- 使用CPU训练：`--device=cpu`

### Q: WandB初始化失败怎么办？
A: 框架会自动处理WandB问题：
- 自动切换到离线模式
- 不影响实验正常运行
- 实验结束后可手动同步数据

### Q: 如何查看实验详细日志？
A: 查看控制台输出，包含：
- 实验配置摘要
- 每轮训练进度
- 客户端数据统计
- 最终结果汇总

---

🎉 **开始你的联邦学习实验之旅吧！** 从简单的MNIST实验开始，逐步探索CLIP多模态联邦学习的强大功能。
