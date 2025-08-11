# 数据使用指南

本文档介绍如何在联邦学习框架中使用真实数据集。

## 支持的数据类型

### 1. CSV文件数据集

框架支持从CSV文件加载结构化数据。

#### 使用方法

```python
from data.data_loader import FederatedDataLoader

# 创建数据加载器
data_loader = FederatedDataLoader(num_clients=3, batch_size=32)

# 加载CSV数据
client_dataloaders, test_dataloader = data_loader.load_csv_data(
    csv_path="./data/your_dataset.csv",
    target_column="target",           # 目标列名
    data_type="classification",       # "classification" 或 "regression"
    test_size=0.2,                   # 测试集比例
    random_state=42,                 # 随机种子
    iid=True                         # 是否使用IID数据分布
)
```

#### CSV文件要求

- 第一行应为列名
- 目标列包含分类标签或回归值
- 缺失值会被自动处理（填充为0）
- 分类标签会自动编码为数字

#### 示例CSV格式

```csv
feature1,feature2,feature3,target
1.2,3.4,5.6,class_A
2.1,4.3,6.5,class_B
3.0,5.2,7.4,class_A
...
```

### 2. 图像数据集

支持从文件夹结构加载图像数据，每个子文件夹代表一个类别。

#### 使用方法

```python
# 加载图像数据
client_dataloaders, test_dataloader = data_loader.load_image_data(
    data_dir="./data/images/",       # 图像文件夹路径
    data_type="classification",      # 数据类型
    test_size=0.2,                  # 测试集比例
    random_state=42,                # 随机种子
    iid=True,                       # 数据分布
    image_size=(32, 32)             # 图像尺寸
)
```

#### 文件夹结构示例

```
data/images/
├── class_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class_2/
│   ├── image3.jpg
│   ├── image4.jpg
│   └── ...
└── class_3/
    ├── image5.jpg
    └── ...
```

### 3. 内置数据集

支持常用的机器学习数据集，如MNIST、CIFAR-10等。

#### 使用方法

```python
# 加载内置数据集
client_dataloaders, test_dataloader = data_loader.load_built_in_dataset(
    dataset_name="mnist",           # "mnist", "cifar10", "fashion_mnist"
    data_type="classification",     # 数据类型
    random_state=42,               # 随机种子
    iid=False                      # Non-IID分布更符合实际场景
)
```

#### 支持的内置数据集

- **MNIST**: 手写数字识别数据集
- **CIFAR-10**: 自然图像分类数据集
- **Fashion-MNIST**: 时尚用品图像分类数据集

## 数据分布模式

### IID分布 (Independent and Identically Distributed)

数据在各客户端间随机分布，每个客户端的数据分布相似。

```python
iid=True  # 使用IID分布
```

**特点:**
- 各客户端数据分布相似
- 训练较为稳定
- 不太符合实际联邦学习场景

### Non-IID分布

数据在各客户端间非均匀分布，更符合实际应用场景。

```python
iid=False  # 使用Non-IID分布
```

**特点:**
- 各客户端数据分布不同
- 更符合真实联邦学习场景
- 训练更具挑战性
- 需要更好的聚合策略

## 示例数据集

框架提供了几个示例数据集的自动下载功能：

### 下载示例数据集

```python
from data.data_loader import download_sample_data

# 下载鸢尾花数据集
iris_path = download_sample_data("iris")

# 下载葡萄酒数据集
wine_path = download_sample_data("wine")

# 下载波士顿房价数据集（回归）
boston_path = download_sample_data("boston")
```

### 可用的示例数据集

| 数据集名称 | 类型 | 描述 | 目标列 |
|-----------|------|------|--------|
| iris | 分类 | 鸢尾花数据集 | species |
| wine | 分类 | 葡萄酒分类数据集 | class |
| boston | 回归 | 波士顿房价数据集 | medv |

## 数据预处理

框架自动进行以下预处理步骤：

### 特征预处理
- **缺失值处理**: 用0填充NaN值
- **标准化**: 使用StandardScaler进行特征标准化
- **类型转换**: 自动转换为适当的数据类型

### 标签预处理
- **分类标签编码**: 自动将文本标签编码为数字
- **回归标签**: 保持原始数值格式

## 完整示例

### 基本CSV数据集示例

```python
from data.data_loader import FederatedDataLoader, download_sample_data
from core.client import FederatedClient
from core.server import FederatedServer
from models.base import SimpleClassificationModel
from communication.local import LocalCommunication
from aggregation.federated_avg import FederatedAveraging

# 1. 下载并加载数据
data_path = download_sample_data("iris")
data_loader = FederatedDataLoader(num_clients=3, batch_size=16)

client_dataloaders, test_dataloader = data_loader.load_csv_data(
    csv_path=data_path,
    target_column="species",
    data_type="classification",
    test_size=0.2,
    random_state=42,
    iid=True
)

# 2. 获取数据维度
sample_batch = next(iter(client_dataloaders[0]))
n_features = sample_batch[0].shape[1]
n_classes = len(torch.unique(sample_batch[1]))

# 3. 创建模型和服务器
global_model = SimpleClassificationModel(n_features, n_classes, 0.01)
server = FederatedServer(global_model, FederatedAveraging())

# 4. 创建客户端
clients = []
communication = LocalCommunication()

for i in range(3):
    client_id = f"client_{i}"
    client_model = SimpleClassificationModel(n_features, n_classes, 0.01)
    client = FederatedClient(
        client_id=client_id,
        model=client_model,
        data_loader=client_dataloaders[i],
        epochs=1,
        learning_rate=0.01
    )
    clients.append(client)
    communication.register_client(client_id)

# 5. 执行联邦学习
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
    
    # 评估
    eval_results = server.evaluate_with_dataloader(test_dataloader)
    print(f"Round {round_num}: {eval_results}")
    
    communication.clear_buffers()
```

## 性能优化建议

### 1. 批次大小选择
- 较小的批次大小 (16-32) 适合小数据集
- 较大的批次大小 (64-128) 适合大数据集
- 考虑客户端的计算能力

### 2. 数据分布选择
- 初期实验使用IID分布
- 实际部署使用Non-IID分布
- 根据应用场景调整分布策略

### 3. 客户端数量
- 2-10个客户端适合实验环境
- 实际应用可扩展到更多客户端
- 考虑通信开销和聚合复杂度

## 故障排除

### 常见问题

1. **数据加载失败**
   - 检查文件路径是否正确
   - 确认CSV文件格式是否符合要求
   - 检查目标列名是否存在

2. **内存不足**
   - 减小批次大小
   - 减少客户端数量
   - 使用数据采样

3. **维度不匹配**
   - 确认所有客户端使用相同的特征数量
   - 检查数据预处理是否一致

### 调试建议

1. **数据统计检查**
```python
# 获取数据统计信息
stats = data_loader.get_data_statistics(client_dataloaders)
print(f"数据统计: {stats}")
```

2. **客户端数据检查**
```python
# 检查客户端数据分布
for i, dataloader in enumerate(client_dataloaders):
    print(f"客户端 {i}: {len(dataloader.dataset)} 样本")
```

3. **模型参数检查**
```python
# 检查模型参数
params = model.get_parameters()
for name, param in params.items():
    print(f"{name}: {param.shape}")
```
