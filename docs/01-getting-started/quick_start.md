# 快速开始

本指南将帮助您快速上手联邦学习框架，目前支持 MNIST 数据集的联邦学习实验。

## 安装

### 1. 克隆项目
```bash
git clone <your-repo-url>
cd myWork
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

所需依赖包括：
- numpy>=1.21.0
- pandas>=1.3.0  
- torch>=1.9.0
- torchvision>=0.10.0
- scikit-learn>=1.0.0
- matplotlib>=3.3.0
- tqdm>=4.62.0
- wandb>=0.15.0
- PyYAML>=6.0

### 3. 验证安装
安装完成后可以直接运行实验，系统会自动下载所需的数据集。

## 第一个联邦学习实验

### 使用 MNIST 数据集

项目目前支持 MNIST 手写数字识别的联邦学习实验，使用 CNN 模型：

```bash
# 基本的 CNN 联邦学习实验
python run_experiment.py --config configs/mnist.yaml
```

这将：
1. 自动下载 MNIST 数据集到 `/home/zzm/dataset/` 目录
2. 创建 10 个客户端，每轮选择所有客户端参与训练
3. 使用 CNN 模型进行 10 轮联邦学习训练
4. 使用 IID 数据分布（数据随机分配给各客户端）
5. 输出每轮的训练结果和准确率

### 自定义实验参数

```bash
# 使用命令行参数覆盖配置文件设置
python run_experiment.py --config configs/mnist.yaml \
    --rounds 10 \
    --num-clients 5 \
    --local-epochs 5 \
    --learning-rate 0.001 \
    --batch-size 64

# 快速测试（少轮次、少客户端）
python run_experiment.py --config configs/default.yaml \
    --rounds 2 \
    --num-clients 2
```

### 其他可用配置

```bash
# 使用默认配置
python run_experiment.py --config configs/default.yaml

# 使用 CLIP 模型配置（多模态联邦学习）
python run_experiment.py --config configs/clip.yaml
```

## 支持的功能特性

### 模型类型
框架支持以下模型类型：
- `cnn`: CNN 模型（用于图像分类，如 MNIST）
- `linear`: 简单线性模型（用于表格数据）
- `clip`: CLIP 多模态模型（用于视觉-语言任务）

### 数据集
目前支持：
- **MNIST**: 手写数字识别数据集（自动下载）
- 支持 IID 数据分布（数据随机均匀分配给各客户端）

### 优化器
支持多种优化器类型：
- SGD（默认）
- Adam
- AdamW（推荐用于 CLIP 模型）

### 聚合算法
- FedAvg（联邦平均算法）
## 编程方式使用

### 基本使用示例

```python
from data.data_loader import FederatedDataLoader
from utils.experiment_runner import ExperimentRunner
from utils.config_manager import ConfigManager

# 方法1：直接使用数据加载器
data_loader = FederatedDataLoader(num_clients=5, batch_size=32)

# 加载 MNIST 数据集
client_dataloaders, test_dataloader = data_loader.load_mnist_dataset(
    random_state=42,
    samples_per_client=6000  # 可选：每个客户端的固定样本数
)

# 方法2：使用配置文件方式（推荐）
config_manager = ConfigManager()
config = config_manager.load_config('configs/mnist.yaml')
experiment_runner = ExperimentRunner(config)
results = experiment_runner.run_experiment()
```

## 配置选项

### 数据分布
- 目前仅支持 IID 分布（数据随机均匀分配给各客户端）

### 模型配置
支持的模型类型及其配置：
```yaml
model:
  type: "cnn"        # 或 "linear", "clip"
  learning_rate: 0.01
```

### 优化器配置
```yaml
optimizer:
  type: "adamw"          # 或 "sgd", "adam"
  learning_rate: 1e-4
  weight_decay: 0.01
  betas: [0.9, 0.98]     # 适用于 Adam/AdamW
  momentum: 0.9          # 适用于 SGD
```

## 下一步

- 查看 [框架架构](../02-architecture/architecture.md) 了解整体设计
- 阅读 [实验指南](../03-tutorials/experiment_guide.md) 了解如何进行更复杂的实验
- 查看 [开发指南](../04-development/README.md) 学习如何扩展框架
- 了解 [WandB 集成](../04-development/wandb_guide.md) 进行实验跟踪
