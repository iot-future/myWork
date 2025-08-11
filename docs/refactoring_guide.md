# 重构后的代码架构说明

## 概述

经过重构后，原本单一的 `FederatedExperiment` 类被分解为多个专门的模块，使代码更加模块化、易于维护和扩展。

## 模块结构

### 1. `run_experiment.py` - 主入口文件
- **职责**: 协调各个模块，提供简洁的实验流程
- **特点**: 
  - 代码量从305行减少到约50行
  - 清晰的错误处理
  - 单一职责原则

### 2. `utils/config_manager.py` - 配置管理器
- **职责**: 配置文件加载、验证和命令行参数处理
- **功能**:
  - 配置文件加载和验证
  - 命令行参数解析
  - 配置覆盖逻辑
  - 配置完整性检查

### 3. `utils/experiment_runner.py` - 实验运行器
- **职责**: 联邦学习实验的核心执行逻辑
- **功能**:
  - 实验环境设置
  - 数据加载和分配
  - 服务器和客户端初始化
  - 联邦学习轮次执行
  - 模型评估

### 4. `utils/model_factory.py` - 模型工厂
- **职责**: 根据配置创建不同类型的模型
- **特点**:
  - 工厂模式设计
  - 易于扩展新的模型类型
  - 统一的模型创建接口

### 5. `utils/results_handler.py` - 结果处理器
- **职责**: 实验结果的保存、格式化和展示
- **功能**:
  - 结果文件保存
  - 模型文件保存
  - 训练进度格式化
  - 实验摘要生成

## 重构收益

### 1. 代码组织更清晰
- 每个模块都有明确的单一职责
- 降低了代码耦合度
- 提高了代码可读性

### 2. 易于维护和扩展
- 需要修改配置相关功能时，只需修改 `ConfigManager`
- 需要添加新模型时，只需扩展 `ModelFactory`
- 需要改变结果输出格式时，只需修改 `ResultsHandler`

### 3. 更好的错误处理
- 集中化的错误处理逻辑
- 更友好的错误信息
- 更强的配置验证

### 4. 更容易测试
- 每个模块可以独立测试
- 模块间的依赖关系清晰
- 更容易编写单元测试

## 使用方式

使用方式保持不变，兼容原有的所有命令行参数：

```bash
# 使用默认配置
python run_experiment.py

# 使用指定配置文件
python run_experiment.py --config configs/mnist.yaml

# 命令行参数覆盖
python run_experiment.py --config configs/default.yaml --rounds 20 --num-clients 10
```

## 扩展指南

### 添加新的模型类型
在 `utils/model_factory.py` 中添加新的模型创建逻辑：

```python
def create_model(model_config: Dict[str, Any]):
    model_type = model_config['type']
    
    if model_type == 'cnn':
        return CNNModel(learning_rate=learning_rate)
    elif model_type == 'new_model':  # 新增模型
        return NewModel(learning_rate=learning_rate)
    # ...
```

### 添加新的配置验证
在 `utils/config_manager.py` 的 `validate_config` 方法中添加新的验证逻辑。

### 自定义结果输出
在 `utils/results_handler.py` 中修改相关方法来改变结果的保存格式或展示方式。
