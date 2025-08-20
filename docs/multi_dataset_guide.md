# 多数据集联邦学习配置指南

本文档介绍如何配置和使用多数据集联邦学习功能。

## 功能概述

多数据集功能允许您为不同的客户端配置不同的数据集组合：
- 某些客户端只使用 MNIST 数据集
- 某些客户端只使用 CIFAR-10 数据集  
- 某些客户端同时使用多个数据集（如 MNIST + CIFAR-10）

## 配置文件示例

### 多数据集配置 (configs/multi_dataset.yaml)

```yaml
# 实验基本设置
experiment:
  name: "multi_dataset_federated_experiment"
  rounds: 10
  seed: 42

# 客户端设置
client:
  num_clients: 4  # 总客户端数量
  local_epochs: 5
  learning_rate: 0.001
  
  # 客户端数据集配置
  client_datasets:
    client_0:
      - mnist
    client_1:
      - mnist
    client_2:
      - cifar10
    client_3:
      - mnist
      - cifar10  # 多数据集客户端

# 数据设置
data:
  data_dir: "/home/zzm/dataset/"
  batch_size: 32
  
  # 数据集配置
  datasets:
    mnist:
      train: true
    cifar10:
      train: true
```

### 单数据集配置示例 (configs/simple_mnist.yaml)

```yaml
# 实验基本设置
experiment:
  name: "simple_mnist_experiment"
  rounds: 5
  seed: 42

# 客户端设置
client:
  num_clients: 3
  local_epochs: 3
  learning_rate: 0.01
  
  # 客户端数据集配置 - 所有客户端使用相同数据集
  client_datasets:
    client_0:
      - mnist
    client_1:
      - mnist
    client_2:
      - mnist

# 数据设置
data:
  data_dir: "/home/zzm/dataset/"
  batch_size: 64
  
  # 数据集配置
  datasets:
    mnist:
      train: true
```

## 配置说明

### client_datasets 配置项

- `client_0`, `client_1`, ... 对应客户端ID
- 每个客户端可以配置一个或多个数据集
- 数据集名称必须在 `data.datasets` 中定义

### data.datasets 配置项

定义可用的数据集及其参数：
- `mnist`: MNIST数据集配置
- `cifar10`: CIFAR-10数据集配置
- 可以添加其他支持的数据集

## 运行实验

### 使用多数据集配置

```bash
python test_multi_dataset.py --config configs/multi_dataset.yaml
```

### 使用简单MNIST配置

```bash
python test_multi_dataset.py --config configs/simple_mnist.yaml
```

### 使用默认配置

```bash
python main.py --config configs/default.yaml
```

## 必需配置项

所有配置文件必须包含以下配置项：

### client.client_datasets
定义每个客户端使用的数据集列表：
- 键为 `client_0`, `client_1`, ... `client_{n-1}`
- 值为数据集名称列表
- 必须为所有客户端配置数据集

### data.datasets  
定义可用的数据集及其参数：
- 键为数据集名称（如 `mnist`, `cifar10`）
- 值为数据集配置字典
- `client_datasets` 中使用的数据集必须在此处定义

## 客户端数据处理

### 单数据集客户端

直接使用指定数据集的数据加载器。

### 多数据集客户端

系统会自动将多个数据集合并：
- 使用 `torch.utils.data.ConcatDataset` 合并数据集
- 保持数据分布的一致性
- 自动处理批次大小等参数

## 数据分割策略

- 每个数据集在所有使用它的客户端之间均匀分割（IID）
- 使用固定随机种子确保可复现性
- 支持不同客户端使用不同数据集组合

## 注意事项

1. **配置完整性**: 所有客户端必须在 `client_datasets` 中配置
2. **数据集一致性**: `client_datasets` 中使用的数据集必须在 `data.datasets` 中定义
3. **内存使用**: 多数据集客户端会使用更多内存
4. **模型兼容性**: 确保模型能处理不同数据集的输入格式
5. **评估指标**: 当前使用第一个数据集作为测试集

## 扩展支持

要添加新的数据集支持：

1. 在 `data/datasets/` 目录下创建数据集类
2. 在 `data/data_loader.py` 的 `SUPPORTED_DATASETS` 中注册
3. 在配置文件的 `data.datasets` 中添加配置

## 故障排除

### 常见错误

1. **配置文件缺少必需节**: 确保配置文件包含所有必需的配置节
2. **客户端未配置**: 确保为所有客户端都在 `client_datasets` 中配置了数据集
3. **数据集不存在**: 确保客户端使用的数据集在 `data.datasets` 中定义
4. **配置文件格式**: 检查 YAML 文件的缩进和格式是否正确

### 调试建议

1. 查看配置验证的输出信息
2. 使用 `--config` 参数指定配置文件
3. 查看控制台输出的数据统计信息
4. 检查数据目录是否存在且可访问
