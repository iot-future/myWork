# WandB 集成使用指南

## 概述

本项目已集成 Weights & Biases (WandB) 进行实验记录和可视化。WandB 会自动记录：

- **客户端训练指标**：每个客户端的训练损失和准确率
- **全局模型指标**：聚合后全局模型的损失和准确率

## 安装依赖

确保安装了 wandb：

```bash
pip install -r requirements.txt
```

## 配置 WandB

### 1. 初始化 WandB 账户

如果是第一次使用，需要登录 WandB：

```bash
wandb login
```

### 2. 启用 WandB 记录

在配置文件中设置：

```yaml
wandb:
  enabled: true
  project: "your-project-name"
```

## 使用示例

### 基本使用

```bash
python run_experiment.py --config configs/mnist.yaml
```

## 记录的指标

### 客户端指标
- `client/{client_id}/loss`：客户端训练损失
- `client/{client_id}/accuracy`：客户端训练准确率

### 全局指标
- `global/loss`：全局模型损失
- `global/accuracy`：全局模型准确率

## 代码结构

WandB 集成主要在以下文件中实现：

1. **`utils/wandb_logger.py`**：简洁的 WandB 记录函数
2. **`utils/experiment_runner.py`**：集成 WandB 记录到实验流程
3. **`core/client.py`**：修改客户端返回训练指标
4. **`aggregation/federated_avg.py`**：支持新的客户端更新格式

## 特点

- **简洁设计**：使用简单的函数式接口
- **自动记录**：训练过程中自动记录指标，无需手动调用
- **可选启用**：通过配置文件控制，不影响现有功能
- **兼容现有代码**：向后兼容，不破坏现有实验流程

## 日志输出示例

### 成功初始化
```
✓ WandB 初始化成功
...
✓ WandB 记录会话结束
```
