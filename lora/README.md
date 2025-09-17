# LoRA 模块

为联邦学习框架提供 LoRA (Low-Rank Adaptation) 功能支持的完整模块。

## 概述

本模块采用**包装器模式**设计，可以为现有的联邦学习模型（特别是CLIP模型）添加LoRA功能，而无需修改原有代码结构。设计目标是与现有框架完全解耦，提供即插即用的LoRA功能。

## 特性

- 🔧 **即插即用**: 包装器模式，无需修改现有模型代码
- 🎯 **CLIP专用**: 专门为CLIP模型优化的LoRA配置
- 📊 **参数高效**: 大幅减少可训练参数，提高训练效率
- 🔄 **联邦学习兼容**: 完全兼容现有的联邦学习参数管理接口
- 💾 **模型管理**: 提供完整的LoRA模型保存、加载、管理功能
- ⚙️ **配置灵活**: 支持多种预设配置和自定义配置

## 安装依赖

```bash
pip install torch peft transformers
```

## 模块结构

```
lora/
├── __init__.py          # 模块入口
├── config.py            # LoRA配置管理
├── base.py              # LoRA基础包装器
├── clip_wrapper.py      # CLIP专用LoRA包装器
├── factory.py           # LoRA模型工厂和管理器
├── test_lora.py         # 测试脚本
└── README.md            # 本文档
```

## 快速开始

### 1. 基本使用

```python
from lora import LoRAConfig, quick_clip_lora

# 快速创建CLIP LoRA模型
model = quick_clip_lora(
    model_name="openai/clip-vit-base-patch32",
    num_classes=10,
    lora_r=16
)

# 模型训练和使用与原始FederatedCLIPModel完全相同
```

### 2. 包装现有模型

```python
from models.clip import FederatedCLIPModel
from lora.clip_wrapper import CLIPLoRAWrapper
from lora.lora_config import LoRAConfig

# 创建原始联邦CLIP模型
original_model = FederatedCLIPModel(
    model_name="openai/clip-vit-base-patch32",
    num_classes=10
)

# 创建LoRA配置
lora_config = LoRAConfig.for_clip_model(r=16, lora_alpha=32)

# 创建LoRA包装器
lora_wrapper = CLIPLoRAWrapper(
    federated_clip_model=original_model,
    lora_config=lora_config,
    target_components=['encoder']  # 只对编码器应用LoRA
)

# 应用LoRA
lora_wrapper.apply_lora_to_federated_model()

# 现在original_model的接口保持不变，但内部使用LoRA
```

### 3. 配置管理

```python
from lora.lora_config import LoRAConfig
from lora.factory import LoRAConfigManager

# 创建不同的配置
small_config = LoRAConfig.for_clip_model(r=8, lora_alpha=16)
large_config = LoRAConfig.for_clip_model(r=32, lora_alpha=64)

# 使用配置管理器
config_manager = LoRAConfigManager()
config_manager.save_config(small_config, "clip_small")
config_manager.save_config(large_config, "clip_large")

# 加载配置
loaded_config = config_manager.load_config("clip_small")
```

### 4. 模型管理

```python
from lora.factory import LoRAModelManager

# 创建模型管理器
model_manager = LoRAModelManager()

# 保存LoRA模型
model_manager.save_model(lora_wrapper, "my_clip_lora_model")

# 加载LoRA模型
model_manager.load_model("my_clip_lora_model", lora_wrapper)

# 查看模型信息
info = model_manager.get_model_info("my_clip_lora_model")
print(info)
```

## 配置选项

### LoRAConfig 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `r` | 16 | LoRA rank，控制低秩分解的维度 |
| `lora_alpha` | 32 | LoRA scaling 参数 |
| `lora_dropout` | 0.1 | LoRA层的dropout概率 |
| `target_modules` | 自动推断 | 要应用LoRA的模块名称 |
| `bias` | "none" | bias处理方式 |

### 预设配置

```python
from lora.lora_config import LoRAConfig

# CLIP模型配置（不同规模）
small_config = LoRAConfig.for_clip_model(r=8, lora_alpha=16)   # 小规模
medium_config = LoRAConfig.for_clip_model(r=16, lora_alpha=32) # 中等规模
large_config = LoRAConfig.for_clip_model(r=32, lora_alpha=64)  # 大规模

# 分类头配置
classifier_config = LoRAConfig.for_classification_head(r=8, lora_alpha=16)
```

## 联邦学习集成

### 与现有框架的兼容性

LoRA包装器完全兼容现有的联邦学习参数管理接口：

```python
# 获取参数（自动返回LoRA参数或完整参数）
params = model.get_parameters()

# 设置参数
model.set_parameters(params)

# 训练步骤
loss = model.train_step(data, labels)

# 评估
metrics = model.evaluate(data, labels)
```

### 参数传输优化

```python
# 配置为只传输LoRA参数（大幅减少通信开销）
lora_config = LoRAConfig.for_clip_model(r=16)
lora_config.save_only_trainable = True  # 只保存可训练参数

# 参数效率示例：
# 原始CLIP模型: ~151M 参数
# LoRA r=16: ~2.4M 参数 (减少 98.4%)
```

## 高级用法

### 1. 选择性应用LoRA

```python
# 只对编码器应用LoRA
lora_wrapper = CLIPLoRAWrapper(
    federated_clip_model=model,
    target_components=['encoder']
)

# 对编码器和分类头都应用LoRA
lora_wrapper = CLIPLoRAWrapper(
    federated_clip_model=model,
    target_components=['encoder', 'classifier']
)
```

### 2. 梯度检查点

```python
# 启用梯度检查点以节省内存
lora_wrapper.enable_gradient_checkpointing()

# 禁用梯度检查点
lora_wrapper.disable_gradient_checkpointing()
```

### 3. 权重合并

```python
# 将LoRA权重合并到原始模型
merged_model = lora_wrapper.merge_and_unload()
```

### 4. 动态启用/禁用LoRA

```python
# 启用LoRA层
lora_wrapper.enable_lora_layers()

# 禁用LoRA层（使用原始权重）
lora_wrapper.disable_lora_layers()
```

## 性能优势

### 参数效率对比

| 模型 | 总参数 | 可训练参数 | 效率比 |
|------|---------|------------|--------|
| CLIP-ViT-B/32 (完整) | 151M | 151M | 100% |
| CLIP + LoRA (r=8) | 151M | 1.2M | 0.8% |
| CLIP + LoRA (r=16) | 151M | 2.4M | 1.6% |
| CLIP + LoRA (r=32) | 151M | 4.8M | 3.2% |

### 内存和通信优势

- **内存占用**: LoRA只需存储低秩分解矩阵，显著降低内存需求
- **通信开销**: 联邦学习中只需传输LoRA参数，减少98%+的网络传输
- **训练速度**: 更少的参数更新，加速训练过程

## 测试

运行测试脚本验证功能：

```bash
cd lora
python test_lora.py
```

测试内容包括：
- 依赖项检查
- 配置管理测试
- 包装器创建测试
- 工厂功能测试
- CLIP集成测试

## 后续集成步骤

要将LoRA功能集成到现有项目中，建议按以下步骤进行：

### 1. 配置文件集成

在 `configs/` 目录下添加LoRA相关配置：

```yaml
# configs/clip_lora.yaml
model:
  type: "clip_lora"
  model_name: "openai/clip-vit-base-patch32"
  num_classes: 10
  
lora:
  enabled: true
  r: 16
  lora_alpha: 32
  target_components: ["encoder"]
  save_only_trainable: true
```

### 2. 模型工厂集成

在 `utils/model_factory.py` 中添加LoRA模型创建：

```python
def create_model(config):
    if config.model.type == "clip_lora":
        from lora import quick_clip_lora
        return quick_clip_lora(
            model_name=config.model.model_name,
            num_classes=config.model.num_classes,
            lora_r=config.lora.r
        )
    # ... 其他模型类型
```

### 3. 训练脚本修改

现有训练脚本无需修改，LoRA模型与原始模型接口完全兼容。

## 注意事项

1. **依赖管理**: 确保安装了 `torch`, `peft`, `transformers`
2. **内存管理**: 大模型应用LoRA时注意显存使用
3. **配置兼容**: 加载保存的LoRA权重时确保配置兼容
4. **版本兼容**: 建议使用稳定版本的依赖库

## 许可证

本模块遵循项目的整体许可证。