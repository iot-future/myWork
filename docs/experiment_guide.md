# 联邦学习实验运行指南

## 概述

`run_experiment.py` 是联邦学习框架的统一实验入口，支持通过YAML配置文件和命令行参数来配置和运行实验。

## 快速开始

### 1. 使用默认配置运行
```bash
python run_experiment.py
```

### 2. 使用指定配置文件
```bash
python run_experiment.py --config configs/mnist.yaml
```

### 3. 使用命令行参数覆盖配置
```bash
python run_experiment.py --config configs/default.yaml --rounds 10 --num-clients 5 --learning-rate 0.001
```

## 配置文件

### 预设配置文件

- `configs/default.yaml`: 默认配置，适用于基础实验
- `configs/mnist.yaml`: MNIST数据集专用配置
- `configs/non_iid.yaml`: Non-IID数据分布配置

### 配置文件结构

```yaml
experiment:
  name: "实验名称"
  rounds: 10              # 联邦学习轮次
  seed: 42               # 随机种子

client:
  num_clients: 5         # 客户端总数
  selected_clients_per_round: 3  # 每轮选择的客户端数
  local_epochs: 5        # 本地训练轮次
  learning_rate: 0.01    # 客户端学习率

server:
  aggregation_method: "federated_avg"  # 聚合方法

model:
  type: "cnn"           # 模型类型
  learning_rate: 0.01   # 模型学习率

data:
  dataset: "mnist"      # 数据集
  batch_size: 32        # 批大小
  iid: true            # 是否IID分布

evaluation:
  evaluate_every: 1     # 评估频率
```

## 命令行参数

### 基本参数
- `--config, -c`: 配置文件路径 (默认: configs/default.yaml)

### 实验参数覆盖
- `--rounds`: 联邦学习轮次数
- `--num-clients`: 客户端总数
- `--selected-clients`: 每轮选择的客户端数
- `--local-epochs`: 客户端本地训练轮次
- `--learning-rate`: 学习率
- `--batch-size`: 批大小
- `--iid`: 使用IID数据分布
- `--non-iid`: 使用Non-IID数据分布
- `--seed`: 随机种子

## 使用示例

### 1. 快速测试实验
```bash
python run_experiment.py --rounds 3 --num-clients 2 --selected-clients 1
```

### 2. MNIST数据集实验
```bash
python run_experiment.py --config configs/mnist.yaml --rounds 20
```

### 3. Non-IID数据分布实验
```bash
python run_experiment.py --non-iid --rounds 15 --num-clients 8
```

### 4. 自定义学习率实验
```bash
python run_experiment.py --learning-rate 0.001 --batch-size 64
```

### 5. 大规模实验
```bash
python run_experiment.py --config configs/mnist.yaml --num-clients 20 --selected-clients 10 --rounds 50
```

## 输出结果

实验完成后，结果会保存在 `results_dir` 指定的目录中：

- `{experiment_name}_results.yaml`: 包含配置和评估指标的完整结果
- `{experiment_name}_model.pth`: 训练完成的全局模型参数

### 结果文件示例
```yaml
config:
  # 完整的实验配置
metrics:
  - round: 1
    accuracy: 0.8500
    loss: 0.4523
  - round: 2
    accuracy: 0.8750
    loss: 0.3821
```

## 运行实验示例脚本

使用 `examples/run_experiments.py` 来运行多个预设实验：

```bash
python examples/run_experiments.py
```

或者传递参数运行特定实验：
```bash
python examples/run_experiments.py --config configs/mnist.yaml --rounds 5
```

## 注意事项

1. 确保已安装所需依赖: `torch`, `torchvision`, `numpy`, `pyyaml`
2. 首次运行会自动下载MNIST数据集到 `data/` 目录
3. 实验结果会自动创建并保存到 `results/` 目录
4. 命令行参数优先级高于配置文件设置
5. 使用 `--iid` 和 `--non-iid` 参数时是互斥的

## 扩展配置

要添加新的配置选项，只需：

1. 在配置文件中添加新字段
2. 在 `override_config_with_args()` 函数中添加对应的命令行参数处理
3. 在 `FederatedExperiment` 类中使用新配置
