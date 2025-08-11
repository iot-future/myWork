# 数据变换设计模式最佳实践

## 问题背景

在机器学习项目中，同一份数据集往往需要在不同类型的模型中使用，每种模型对输入数据的格式要求不同：

- **线性模型**：需要展平的一维向量 `[batch_size, features]`
- **CNN模型**：需要多维图像格式 `[batch_size, channels, height, width]`
- **RNN模型**：需要序列格式 `[batch_size, sequence_length, features]`

## 设计原则

### 1. 单一职责原则 (Single Responsibility Principle)
- **数据加载器**：负责数据的加载、预处理和格式化
- **模型**：专注于学习算法和推理逻辑

### 2. 开闭原则 (Open/Closed Principle)
- 对扩展开放：新增模型类型时，只需添加相应的数据变换函数
- 对修改封闭：现有模型和数据加载逻辑无需修改

### 3. 依赖倒置原则 (Dependency Inversion Principle)
- 模型依赖于抽象的数据接口，而不是具体的数据格式
- 数据加载器通过变换函数适配不同模型的需求

## 推荐架构

```
原始数据 → 数据变换函数 → 格式化数据 → 模型
```

### 优势分析

#### ✅ 在数据加载器中进行变换（推荐）

**优点：**
1. **解耦合**：模型不需要知道原始数据格式
2. **复用性**：同一模型可以处理不同来源的数据
3. **一致性**：所有数据预处理逻辑集中管理
4. **可测试性**：数据变换逻辑独立，容易单元测试
5. **可扩展性**：新增模型类型时，只需添加变换函数

**实现方式：**
```python
# 为不同模型创建不同的数据加载器
linear_loader = FederatedDataLoader(
    num_clients=3,
    data_transform=DataTransforms.for_linear_model
)

cnn_loader = FederatedDataLoader(
    num_clients=3,
    data_transform=DataTransforms.for_cnn_model
)
```

#### ❌ 在模型中进行变换（不推荐）

**缺点：**
1. **高耦合**：模型需要知道并处理多种数据格式
2. **代码重复**：每个模型都需要实现数据变换逻辑
3. **难以维护**：数据格式变更需要修改所有相关模型
4. **职责不清**：模型既要处理学习逻辑又要处理数据格式

## 实现示例

### 数据变换函数

```python
class DataTransforms:
    @staticmethod
    def for_linear_model(data: np.ndarray) -> torch.Tensor:
        """线性模型：保持展平状态"""
        return torch.FloatTensor(data)
    
    @staticmethod
    def for_cnn_model(data: np.ndarray, channels=1, height=28, width=28) -> torch.Tensor:
        """CNN模型：重新整形为图像格式"""
        if len(data.shape) == 2:
            return torch.FloatTensor(data).view(-1, channels, height, width)
        return torch.FloatTensor(data)
    
    @staticmethod
    def for_rnn_model(data: np.ndarray, sequence_length: int) -> torch.Tensor:
        """RNN模型：整形为序列格式"""
        batch_size, features = data.shape
        features_per_step = features // sequence_length
        return torch.FloatTensor(data).view(batch_size, sequence_length, features_per_step)
```

### 使用方式

```python
# 1. 为线性模型准备数据
linear_data_loader = FederatedDataLoader(
    num_clients=3,
    data_transform=DataTransforms.for_linear_model
)

# 2. 为CNN模型准备数据
cnn_data_loader = FederatedDataLoader(
    num_clients=3,
    data_transform=DataTransforms.for_cnn_model
)

# 3. 模型只需关注业务逻辑
linear_model = SimpleLinearModel(input_dim=784, output_dim=10)
cnn_model = CNNModel()

# 数据已经是模型期望的格式
for data, labels in linear_data_loader:
    linear_model.train_step(data, labels)  # data: [batch, 784]

for data, labels in cnn_data_loader:
    cnn_model.train_step(data, labels)     # data: [batch, 1, 28, 28]
```

## 测试策略

### 1. 数据变换测试
```python
def test_data_transforms():
    # 测试输入输出形状
    input_data = np.random.randn(100, 784)
    
    # 测试线性变换
    linear_output = DataTransforms.for_linear_model(input_data)
    assert linear_output.shape == (100, 784)
    
    # 测试CNN变换
    cnn_output = DataTransforms.for_cnn_model(input_data)
    assert cnn_output.shape == (100, 1, 28, 28)
```

### 2. 集成测试
```python
def test_model_integration():
    # 确保变换后的数据能被模型正确处理
    data_loader = FederatedDataLoader(transform=DataTransforms.for_cnn_model)
    model = CNNModel()
    
    for data, labels in data_loader:
        loss = model.train_step(data, labels)
        assert loss is not None
```

## 扩展性示例

当需要支持新的模型类型时：

```python
# 1. 添加新的变换函数
class DataTransforms:
    @staticmethod
    def for_transformer_model(data: np.ndarray, patch_size: int = 4) -> torch.Tensor:
        """Transformer模型：将图像分割为patches"""
        # 实现patch化逻辑
        pass

# 2. 使用新变换
transformer_loader = FederatedDataLoader(
    data_transform=DataTransforms.for_transformer_model
)

# 3. 现有代码无需修改
```

## 总结

**推荐做法：**
- 在数据加载器中进行数据变换
- 使用策略模式设计变换函数
- 保持模型专注于学习逻辑
- 通过依赖注入传递变换函数

这种设计模式符合SOLID原则，提高了代码的可维护性、可测试性和可扩展性，是机器学习项目中数据处理的最佳实践。
