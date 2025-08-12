# 新增模型快速参考指南

这是一个精简版的模型添加指南，适合有经验的开发者快速参考。

## 🚀 快速步骤

### 1. 创建模型文件
```python
# models/your_model.py
from core.base import BaseModel

class YourModel(BaseModel):
    def __init__(self, optimizer_config=None):
        super().__init__(optimizer_config)
        self.model = nn.Sequential(...)  # 定义架构
        self.create_optimizer(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()
    
    def get_parameters(self): 
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
    
    def set_parameters(self, params):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params:
                    param.data.copy_(params[name])
    
    def train_step(self, data, labels):
        # 实现训练逻辑
        pass
    
    def evaluate(self, data, labels):
        # 实现评估逻辑  
        pass
```

### 2. 更新模型工厂
```python
# utils/model_factory.py
from models.your_model import YourModel  # 添加导入

# 在 create_model 方法中添加：
elif model_type == 'your_model':
    return YourModel(optimizer_config=optimizer_config)

# 在 get_supported_models 中添加：
return [..., 'your_model']
```

### 3. 更新导出
```python
# models/__init__.py
from .your_model import YourModel
__all__ = [..., 'YourModel']
```

### 4. 创建配置文件
```yaml
# configs/your_model.yaml
experiment:
  name: "your_model_experiment"
  rounds: 50

model:
  type: "your_model"
  # 添加模型参数

optimizer:
  type: "adam"
  learning_rate: 0.001
```

### 5. 测试验证
```python
# 快速测试
from utils.model_factory import ModelFactory
model = ModelFactory.create_model({'type': 'your_model'})
print("✓ 模型创建成功")
```

## ✅ 检查清单

- [ ] 继承了 `BaseModel`
- [ ] 实现了所有抽象方法（4个）
- [ ] 在模型工厂中添加了创建逻辑
- [ ] 更新了 `__init__.py` 导出
- [ ] 创建了配置文件
- [ ] 基本功能测试通过

## 🎯 必须实现的方法

| 方法名 | 用途 | 返回值 |
|--------|------|--------|
| `get_parameters()` | 联邦学习参数获取 | Dict[str, Tensor] |
| `set_parameters()` | 联邦学习参数设置 | None |
| `train_step()` | 单步训练 | float (loss) |
| `evaluate()` | 模型评估 | Dict[str, float] |

## ⚠️ 常见陷阱

1. **忘记调用 `super().__init__()`**
2. **没有提供默认优化器**
3. **参数名不匹配导致 set_parameters 失败**
4. **忘记在工厂类中添加模型类型**

## 📝 模型模板

复制以下模板快速开始：

```python
# models/template.py
import torch
import torch.nn as nn
from core.base import BaseModel

class TemplateModel(BaseModel):
    def __init__(self, optimizer_config=None):
        super().__init__(optimizer_config)
        
        # 定义模型
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
        # 创建优化器
        self.create_optimizer(self.model.parameters())
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters())
        
        self.criterion = nn.CrossEntropyLoss()
    
    def get_parameters(self):
        return {n: p.data.clone() for n, p in self.model.named_parameters()}
    
    def set_parameters(self, params):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in params: p.data.copy_(params[n])
    
    def train_step(self, data, labels):
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(data)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def evaluate(self, data, labels):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            _, pred = torch.max(outputs, 1)
            acc = (pred == labels).float().mean().item()
        return {'loss': loss.item(), 'accuracy': acc}
```

只需修改模型架构部分即可！
