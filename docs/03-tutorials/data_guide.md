# 数据使用指南

本文档介绍联邦学习框架中的数据加载和处理功能。

## 支持的数据集

框架目前支持MNIST手写数字识别数据集，这是联邦学习研究中最常用的基准数据集。

### MNIST数据集加载

#### 基本用法

```python
from data.data_loader import FederatedDataLoader

# 创建数据加载器
data_loader = FederatedDataLoader(num_clients=3, batch_size=32)

# 加载MNIST数据集
client_dataloaders, test_dataloader = data_loader.load_mnist_dataset(
    random_state=42
)
```

#### 基线实验模式

支持每个客户端固定样本数的设置，用于重现经典联邦学习论文的实验配置：

```python
# 每个客户端使用固定数量的样本
client_dataloaders, test_dataloader = data_loader.load_mnist_dataset(
    random_state=42,
    samples_per_client=600  # 每个客户端600个样本
)
```

## 数据变换

框架支持多种数据变换，适配不同的模型架构：

### 数据变换类型

#### 线性模型变换（默认）
```python
from data.data_loader import DataTransforms

# 默认变换，将数据保持为展平格式
data_loader = FederatedDataLoader(
    num_clients=3,
    data_transform=DataTransforms.for_linear_model
)
```

#### CNN模型变换
```python
# 将展平数据重新整形为图像格式 (batch, 1, 28, 28)
data_loader = FederatedDataLoader(
    num_clients=3,
    data_transform=DataTransforms.for_cnn_model
)
```

#### RNN模型变换
```python
# 将数据整形为序列格式
data_loader = FederatedDataLoader(
    num_clients=3,
    data_transform=lambda data: DataTransforms.for_rnn_model(data, sequence_length=28)
)
```

## 数据分布

框架支持IID（独立同分布）数据分布模式：

### IID分布
- 数据在各客户端间随机均匀分布
- 每个客户端的数据分布相似
- 适合初期实验和算法验证

### 分布模式

1. **标准IID分布**: 将所有训练数据随机分配给各客户端
2. **基线IID分布**: 每个客户端使用固定数量的样本（用于重现论文实验）

## 数据预处理

### MNIST预处理流程
1. **标准化**: 将像素值归一化到 [-1, 1] 范围
2. **张量转换**: 转换为PyTorch张量格式
3. **数据展平**: 将28×28图像展平为784维向量（可选，取决于模型类型）
4. **分类标签**: 自动转换为LongTensor格式

## 完整使用示例

```python
from data.data_loader import FederatedDataLoader, DataTransforms
from core.client import FederatedClient
from core.server import FederatedServer
from models.base import SimpleClassificationModel
from communication.local import LocalCommunication
from aggregation.federated_avg import FederatedAveraging

# 1. 创建数据加载器（适配线性模型）
data_loader = FederatedDataLoader(
    num_clients=3,
    batch_size=32,
    data_transform=DataTransforms.for_linear_model
)

# 2. 加载MNIST数据集
client_dataloaders, test_dataloader = data_loader.load_mnist_dataset(
    random_state=42,
    samples_per_client=600  # 可选：每个客户端固定样本数
)

# 3. 获取数据维度信息
sample_batch = next(iter(client_dataloaders[0]))
n_features = sample_batch[0].shape[1]  # 784 (28*28)
n_classes = 10  # MNIST有10个数字类别

# 4. 创建全局模型和服务器
global_model = SimpleClassificationModel(n_features, n_classes)
server = FederatedServer(global_model, FederatedAveraging())

# 5. 创建客户端
clients = []
communication = LocalCommunication()

for i in range(3):
    client_id = f"client_{i}"
    client_model = SimpleClassificationModel(n_features, n_classes)
    client = FederatedClient(
        client_id=client_id,
        model=client_model,
        data_loader=client_dataloaders[i]
    )
    clients.append(client)
    communication.register_client(client_id)

# 6. 执行联邦学习
for round_num in range(1, 6):
    # 获取全局模型参数
    global_params = server.send_global_model()
    communication.broadcast_to_clients(global_params)
    
    # 客户端训练
    client_updates = []
    for client in clients:
        messages = communication.receive_from_server(client.client_id)
        latest_params = messages[-1] if messages else global_params
        updated_params = client.train(latest_params)
        communication.send_to_server(client.client_id, updated_params)
        client_updates.append(updated_params)
    
    # 服务器聚合
    server.aggregate(client_updates)
    
    # 评估模型
    eval_results = server.evaluate_with_dataloader(test_dataloader)
    print(f"Round {round_num}: {eval_results}")
    
    communication.clear_buffers()
```

## 数据统计与监控

### 获取数据统计信息

```python
# 获取客户端数据分布统计
stats = data_loader.get_data_statistics(client_dataloaders)
print(f"数据统计信息:")
print(f"- 总客户端数: {stats['total_clients']}")
print(f"- 总样本数: {stats['total_samples']}")
print(f"- 平均每客户端样本数: {stats['avg_client_size']:.0f}")
print(f"- 最小客户端样本数: {stats['min_client_size']}")
print(f"- 最大客户端样本数: {stats['max_client_size']}")

# 获取详细的客户端数据信息
client_info = data_loader.get_client_data_info()
for client_id, info in client_info.items():
    print(f"{client_id}: {info['size']} 样本")
```

## 配置建议

### 客户端数量选择
- **实验环境**: 2-5个客户端，便于调试
- **性能测试**: 10-20个客户端，测试聚合算法性能
- **大规模模拟**: 50+个客户端，接近真实场景

### 批次大小选择
- **小数据集**: 16-32，适合快速迭代
- **大数据集**: 64-128，提高训练效率
- **内存受限**: 8-16，避免内存溢出

### 每客户端样本数
- **基线实验**: 600样本（经典设置）
- **不平衡实验**: 100-1000样本（模拟真实场景）
- **大规模实验**: 使用全部数据平均分配

## 故障排除

### 常见问题

1. **数据下载失败**
   - 确保网络连接正常
   - 检查MNIST数据集下载路径权限
   - 手动下载数据集到指定路径

2. **内存不足**
   - 减小批次大小 (`batch_size`)
   - 减少客户端数量 (`num_clients`)
   - 使用固定样本数模式 (`samples_per_client`)

3. **数据形状不匹配**
   - 检查数据变换函数是否正确
   - 确认模型输入维度与数据维度匹配
   - 验证标签格式（分类用LongTensor，回归用FloatTensor）

### 调试技巧

```python
# 1. 检查数据加载器
for i, dataloader in enumerate(client_dataloaders):
    print(f"客户端 {i}: {len(dataloader.dataset)} 样本")
    sample_data, sample_label = next(iter(dataloader))
    print(f"  数据形状: {sample_data.shape}")
    print(f"  标签形状: {sample_label.shape}")
    print(f"  标签类型: {sample_label.dtype}")

# 2. 检查数据范围
sample_batch = next(iter(client_dataloaders[0]))
data, labels = sample_batch
print(f"数据范围: [{data.min():.3f}, {data.max():.3f}]")
print(f"标签范围: [{labels.min()}, {labels.max()}]")

# 3. 验证数据一致性
print(f"特征维度: {data.shape[1]}")
print(f"类别数: {len(torch.unique(labels))}")
```
