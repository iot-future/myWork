# CLIP模型LoRA微调使用指南

本指南介绍如何在联邦学习框架中使用LoRA微调CLIP模型。

## 概述

LoRA (Low-Rank Adaptation) 是一种参数高效的微调方法，通过在预训练模型的线性层中添加低秩分解矩阵来实现微调，大大减少了需要训练的参数数量。

## 功能特性

- **参数高效**：只训练LoRA参数，大幅减少训练参数量
- **内存友好**：降低GPU内存需求
- **快速收敛**：LoRA微调通常比全参数微调收敛更快
- **联邦学习兼容**：完全兼容现有的联邦学习框架

## 配置使用

### 1. 启用LoRA

在配置文件中启用LoRA功能：

```yaml
# configs/clip_lora.yaml
model:
  type: "clip"
  model_name: "openai/clip-vit-base-patch32"
  
  # LoRA微调配置
  lora:
    enabled: true                    # 启用LoRA微调
    r: 8                            # LoRA rank (1-64)
    lora_alpha: 16                  # LoRA scaling参数
    lora_dropout: 0.05              # LoRA dropout概率
    target_modules:
      vision_encoder: ["q_proj", "v_proj", "k_proj", "out_proj"]
      text_encoder: ["q_proj", "v_proj", "k_proj", "out_proj"]
```

### 2. 调整训练参数

LoRA微调建议的参数设置：

```yaml
client:
  local_epochs: 3                   # LoRA可以使用更多本地轮数
  learning_rate: 0.0001            # LoRA使用稍高的学习率

optimizer:
  learning_rate: 1e-4              # LoRA微调学习率
  weight_decay: 0.01               # 较小的权重衰减

data:
  batch_size: 32                   # LoRA可以使用较大批次
```

## 代码使用

### 1. 创建LoRA模型

```python
from models.clip import create_clip_model

# 从配置文件创建模型
config = {
    'model_name': 'openai/clip-vit-base-patch32',
    'num_classes': 10,
    'lora': {
        'enabled': True,
        'r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.05,
        'target_modules': {
            'vision_encoder': ['q_proj', 'v_proj', 'k_proj', 'out_proj']
        }
    }
}

model = create_clip_model(config)
```

### 2. 检查LoRA状态

```python
# 检查LoRA是否启用
print(f"LoRA enabled: {model.is_lora_enabled()}")

# 获取LoRA信息
lora_info = model.get_lora_info()
print(f"LoRA trainable parameters: {lora_info['trainable_parameters']}")

# 获取模型摘要
summary = model.get_model_summary()
print(f"Total parameters: {summary['total_parameters']:,}")
print(f"Trainable parameters: {summary['trainable_parameters']:,}")
```

### 3. 联邦学习参数管理

```python
# 获取LoRA参数（用于联邦学习聚合）
lora_params = model.get_parameters()

# 设置LoRA参数（用于模型更新）
model.set_parameters(aggregated_params)
```

## LoRA参数说明

### rank (r)
- **范围**：1-64
- **作用**：控制LoRA矩阵的秩，影响参数量和表达能力
- **建议**：较小的值（4-16）适合大多数任务

### lora_alpha
- **范围**：通常为rank的1-2倍
- **作用**：缩放LoRA输出的强度
- **建议**：一般设为rank的2倍

### lora_dropout
- **范围**：0.0-0.3
- **作用**：LoRA层的dropout概率，防止过拟合
- **建议**：0.05-0.1适合大多数情况

### target_modules
- **选择**：选择要应用LoRA的模块
- **建议**：attention层（q_proj, k_proj, v_proj）是最有效的目标

## 性能对比

| 方法 | 参数量 | 内存占用 | 训练速度 | 性能 |
|------|--------|----------|----------|------|
| 全参数微调 | 100% | 高 | 慢 | 基准 |
| LoRA (r=8) | ~1% | 低 | 快 | 95-99% |
| LoRA (r=16) | ~2% | 低 | 快 | 97-100% |

## 故障排除

### 常见问题

1. **导入错误**
   ```
   ImportError: cannot import 'CLIPLoRAWrapper'
   ```
   解决：确保安装了peft库：`pip install peft`

2. **CUDA内存不足**
   - 减小batch_size
   - 降低LoRA rank
   - 使用梯度累积

3. **训练不稳定**
   - 降低learning_rate
   - 增加lora_dropout
   - 检查数据预处理

### 调试技巧

```python
# 检查LoRA参数
def debug_lora_params(model):
    if model.is_lora_enabled():
        lora_params = model.get_parameters()
        print(f"LoRA parameter count: {len(lora_params)}")
        for name in list(lora_params.keys())[:5]:  # 显示前5个参数
            print(f"  {name}: {lora_params[name].shape}")
    else:
        print("LoRA not enabled")

debug_lora_params(model)
```

## 最佳实践

1. **从小的rank开始**：先尝试r=4或8，再根据需要增加
2. **监控训练指标**：LoRA微调通常收敛更快
3. **合理设置学习率**：LoRA可以使用比全参数微调更高的学习率
4. **保存LoRA权重**：只保存LoRA参数可以大大减少模型大小
5. **实验不同目标模块**：某些任务可能只需要微调特定层

## 扩展功能

框架还支持以下高级功能：

- 动态rank调整
- 分层LoRA配置
- LoRA权重合并
- 多任务LoRA适配

详细信息请参考相关文档或代码注释。