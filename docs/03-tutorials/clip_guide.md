# CLIP模型使用指南

本文档介绍如何在联邦学习框架中使用CLIP（Contrastive Language-Image Pre-training）模型。

## 概述

CLIP模型是一个多模态深度学习模型，能够理解图像和文本之间的关系。在联邦学习场景中，CLIP模型特别适用于：

- **多模态数据的联邦学习**：处理同时包含图像和文本的分布式数据
- **零样本分类**：无需额外训练即可对新类别进行分类
- **跨模态检索**：实现图像到文本或文本到图像的检索
- **表示学习**：学习通用的视觉和语言表示

## 模型类型

### CLIPModel（完整版）

完整的CLIP实现，包含：
- **视觉编码器**：基于Vision Transformer (ViT)
- **文本编码器**：基于Transformer
- **对比学习**：通过InfoNCE损失进行训练

**特点**：
- 高精度，适合大规模任务
- 支持复杂的多模态理解

## 快速开始

### 1. 配置文件

#### CLIP配置（configs/clip.yaml）

```yaml
model:
  type: "clip"
  img_size: 224
  patch_size: 32
  in_channels: 3
  vocab_size: 50000
  max_text_len: 77
  d_model: 512
  n_layers: 12
  n_heads: 8
  temperature: 0.07

optimizer:
  type: "adamw"
  learning_rate: 1e-4
  weight_decay: 0.01
```

### 2. 运行实验

```bash
# 使用CLIP模型
python main.py --config configs/clip.yaml
```

### 3. 代码示例

```python
from utils.model_factory import ModelFactory

# 创建CLIP模型
model_config = {
    'type': 'clip',
    'd_model': 512
}

optimizer_config = {
    'type': 'adamw',
    'learning_rate': 1e-4
}

model = ModelFactory.create_model(model_config, optimizer_config)

# 训练
loss = model.train_step((images, text_tokens))

# 评估
metrics = model.evaluate((images, text_tokens))

# 零样本分类
probabilities = model.zero_shot_classify(images, class_text_features)
```

## 数据格式

### 输入数据

CLIP模型期望的输入数据格式：

```python
# 图像：张量格式 [batch_size, channels, height, width]
images = torch.randn(32, 3, 224, 224)  # 32张RGB图像

# 文本：token序列 [batch_size, sequence_length]
text_tokens = torch.randint(0, 10000, (32, 77))  # 32个文本序列
```

### 数据处理

使用`CLIPDataProcessor`类处理数据：

```python
from examples.clip_example import CLIPDataProcessor

processor = CLIPDataProcessor(img_size=224, max_text_len=77)

# 处理图像-文本对
image_text_pairs = [
    (image1, "a red car"),
    (image2, "a black cat"),
    # ...
]

images_batch, texts_batch = processor.create_batch(image_text_pairs)
```

## 核心功能

### 1. 多模态表示学习

```python
# 编码图像和文本
image_features = model.encode_image(images)      # [batch_size, embed_dim]
text_features = model.encode_text(text_tokens)   # [batch_size, embed_dim]

# 计算相似度
similarity = torch.matmul(image_features, text_features.t())
```

### 2. 零样本分类

```python
# 定义类别描述
class_descriptions = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a car"
]

# 预计算类别特征
class_text_features = model.encode_text(class_texts)

# 零样本分类
probabilities = model.zero_shot_classify(test_images, class_text_features)
```

### 3. 联邦学习集成

CLIP模型完全兼容现有的联邦学习框架：

```python
# 获取模型参数
params = model.get_parameters()

# 设置模型参数（用于联邦平均）
model.set_parameters(aggregated_params)

# 本地训练
for epoch in range(local_epochs):
    loss = model.train_step((images, text_tokens))
```

## 配置参数说明

### 模型参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `img_size` | int | 224 | 输入图像尺寸 |
| `patch_size` | int | 32 | ViT的patch大小（仅完整版） |
| `in_channels` | int | 3 | 输入图像通道数 |
| `vocab_size` | int | 50000/10000 | 词汇表大小 |
| `max_text_len` | int | 77/32 | 最大文本序列长度 |
| `d_model` | int | 512 | 模型隐藏维度 |
| `temperature` | float | 0.07 | 对比学习温度参数 |

### 优化器推荐

**CLIP模型**：推荐使用AdamW，学习率1e-4

## 性能说明

CLIP模型具有以下特点：
- 参数量约150M
- 高精度，适合大规模任务
- 支持复杂的多模态理解

## 注意事项

1. **数据预处理**：确保图像和文本数据的预处理一致
2. **批次大小**：CLIP模型对批次大小敏感，建议使用较大的批次
3. **温度参数**：`temperature`参数影响对比学习的难易程度
4. **词汇表**：实际应用中需要使用专门的tokenizer（如BPE）
5. **设备内存**：CLIP模型需要较大的GPU内存

## 扩展示例

详细的使用示例请参考：
- `examples/clip_example.py`：基础使用演示
- `configs/clip.yaml`：CLIP配置

## 相关论文

- Learning Transferable Visual Representations with Natural Language Supervision (Radford et al., 2021)
- Communication-Efficient Learning of Deep Networks from Decentralized Data (McMahan et al., 2017)

## 常见问题

### Q: 如何处理不同语言的文本？
A: 需要构建对应语言的词汇表和tokenizer，或使用多语言预训练的词嵌入。

### Q: 能否处理其他尺寸的图像？
A: 可以通过修改`img_size`参数来适配不同尺寸，但需要相应调整其他参数。

### Q: 如何提高零样本分类的精度？
A: 1) 使用更好的类别描述；2) 增加模型规模；3) 使用更多的训练数据。

### Q: 模型收敛缓慢怎么办？
A: 1) 调整学习率；2) 使用学习率调度器；3) 增加批次大小；4) 使用预训练权重。
