# 快速开始

本指南将帮助您快速上手联邦学习框架，使用真实数据进行联邦学习实验。

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

### 3. 验证安装
```bash
python check_status.py
```

## 第一个联邦学习实验

### 使用CSV数据集示例

运行内置的鸢尾花数据集示例：

```bash
python examples/simple_example.py --example csv
```

这将：
1. 自动下载鸢尾花数据集
2. 创建3个客户端
3. 进行5轮联邦学习训练
4. 输出每轮的训练结果

### 使用MNIST数据集示例

运行经典的MNIST手写数字识别（CNN模型）：

```bash
# 基本的CNN联邦学习实验
python run_experiment.py --config configs/mnist.yaml
```

这将：
1. 自动下载MNIST数据集
2. 创建10个客户端，每轮选择5个
3. 使用CNN模型进行20轮联邦学习训练
4. 使用Non-IID数据分布（Dirichlet分布，α=0.3）
5. 输出每轮的训练结果和准确率

### 自定义实验参数

```bash
# 使用命令行参数覆盖配置文件设置
python run_experiment.py --config configs/mnist.yaml \
    --rounds 10 \
    --num-clients 5 \
    --selected-clients 3 \
    --local-epochs 5 \
    --learning-rate 0.001

# 快速测试（少轮次、少客户端）
python run_experiment.py --config configs/mnist.yaml \
    --rounds 2 \
    --num-clients 2 \
    --selected-clients 1
```

### 其他实验示例

```bash
# 运行所有示例实验
python examples/run_experiments.py

# 运行独立的简化CNN实验
python examples/simple_federated_mnist_experiment.py
```

## 使用自己的数据

### CSV文件数据

如果您有自己的CSV数据集，可以这样使用：

```bash
python examples/simple_example.py --example custom --csv_path ./data/my_data.csv --target_column target_name
```

#### CSV文件格式要求

您的CSV文件应该具有以下格式：

```csv
feature1,feature2,feature3,target
1.2,3.4,5.6,class_A
2.1,4.3,6.5,class_B
3.0,5.2,7.4,class_A
```

- 第一行为列名
- 最后一列（或指定列）为目标变量
- 特征列应为数值类型
- 目标列可以是分类标签或数值（回归）

### 编程方式使用

```python
from data.data_loader import FederatedDataLoader

# 创建数据加载器
data_loader = FederatedDataLoader(num_clients=3, batch_size=32)

# 加载您的数据
client_dataloaders, test_dataloader = data_loader.load_csv_data(
    csv_path="./data/your_dataset.csv",
    target_column="target_column_name",
    data_type="classification",  # 或 "regression"
    test_size=0.2,              # 测试集比例
    random_state=42,            # 随机种子
    iid=True                    # 数据分布模式
)
```
    )
```

### 2. 运行联邦学习
```python
server = fl_setup["server"]
clients = fl_setup["clients"]
communication = fl_setup["communication"]
test_data, test_labels = fl_setup["test_data"]

# 训练循环
for round_num in range(1, num_rounds + 1):
    # 获取全局模型参数
    global_params = server.send_global_model()
    
    # 广播到客户端
    communication.broadcast_to_clients(global_params)
    
    # 客户端训练
    client_updates = []
    for client in clients:
        messages = communication.receive_from_server(client.client_id)
        latest_params = messages[-1] if messages else global_params
        updated_params = client.train(latest_params)
        client_updates.append(updated_params)
        communication.send_to_server(client.client_id, updated_params)
    
    # 服务器聚合
    server.aggregate(client_updates)
    
    # 评估
    eval_results = server.evaluate_global_model(test_data, test_labels)
    print(f"Round {round_num}: {eval_results}")
    
    communication.clear_buffers()
```

## 配置选项

### 数据分布
- `iid=True`: 所有客户端的数据分布相同
- `iid=False`: 非独立同分布，每个客户端只有部分类别的数据

### 任务类型
- `data_type="classification"`: 分类任务
- `data_type="regression"`: 回归任务

### 模型类型
框架提供两种内置模型：
- `SimpleClassificationModel`: 用于分类任务
- `SimpleLinearModel`: 用于回归任务

## 下一步

- 查看 [框架架构](architecture.md) 了解整体设计
- 阅读 [核心模块](modules/README.md) 了解各模块功能
- 查看 [扩展指南](extensions/README.md) 学习如何扩展框架
