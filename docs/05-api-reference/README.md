# API参考

完整的API文档和技术参考手册，提供详细的接口信息和使用说明。

## 📁 文档列表

### 核心模块API
- **[核心模块](./modules/)** - 框架核心组件的详细API文档
  - [客户端API](./modules/client.md) - FederatedClient类的完整接口
  - [服务器API](./modules/server.md) - FederatedServer类的完整接口

### 快速参考
- **[模型参考](./quick_model_reference.md)** - 所有可用模型的快速查询手册

## 🎯 使用目标

本API参考文档旨在为以下用户提供帮助：
- **开发者**: 查阅接口定义和参数说明
- **集成者**: 了解如何调用框架API
- **维护者**: 理解代码结构和依赖关系

## 📚 API分类

### 核心API
```
🏗️ 核心组件
├── Client API - 客户端接口和方法
├── Server API - 服务器接口和方法
├── Base Classes - 抽象基类定义
└── Communication - 通信接口规范
```

### 模型API
```
🧠 模型系统
├── BaseModel - 模型基类接口
├── CNN Models - 卷积神经网络模型
├── Linear Models - 线性模型
└── Custom Models - 自定义模型接口
```

### 工具API
```
🛠️ 工具组件
├── DataLoader - 数据加载器
├── ConfigManager - 配置管理器
├── Logger - 日志系统
└── ModelFactory - 模型工厂
```

## 📋 使用指南

### 查找API
1. **按功能查找**: 根据需要实现的功能选择对应模块
2. **按组件查找**: 直接查看特定组件的API文档
3. **快速参考**: 使用快速参考手册查找常用接口

### 阅读格式
每个API文档包含：
- **类/函数签名**: 完整的接口定义
- **参数说明**: 详细的参数类型和含义
- **返回值**: 返回值类型和说明
- **使用示例**: 具体的代码示例
- **异常处理**: 可能抛出的异常

### 版本兼容性
- API文档与代码同步更新
- 标注了版本变更信息
- 提供迁移指南（如有重大变更）

## 🔍 快速导航

### 常用API
| 组件 | 主要接口 | 用途 |
|------|----------|------|
| FederatedClient | `train()`, `get_model_params()` | 客户端训练和参数获取 |
| FederatedServer | `aggregate()`, `broadcast()` | 服务器聚合和分发 |
| DataLoader | `load_dataset()`, `create_dataloaders()` | 数据加载和分割 |
| ModelFactory | `create_model()` | 模型实例化 |

### 扩展接口
| 接口 | 继承自 | 用于扩展 |
|------|--------|----------|
| BaseModel | torch.nn.Module | 自定义模型 |
| BaseClient | ABC | 自定义客户端 |
| BaseServer | ABC | 自定义服务器 |
| BaseCommunication | ABC | 自定义通信 |

## 📖 使用建议

### 开发流程
1. **查阅相关API** - 确定需要使用的接口
2. **查看示例代码** - 理解API的使用方式
3. **编写测试代码** - 验证API调用的正确性
4. **集成到项目** - 将API集成到实际项目中

### 最佳实践
- 始终检查API的返回值
- 适当处理可能的异常
- 遵循API的调用约定
- 及时更新到最新版本

## ➡️ 相关资源

- [架构设计](../02-architecture/) - 了解API的设计背景
- [开发指南](../04-development/) - 学习如何扩展API
- [使用教程](../03-tutorials/) - API的实际应用示例

## 🆘 获取帮助

- 每个API都有详细的docstring
- 查看对应的单元测试了解用法
- 参考现有代码中的API调用示例
- 遇到问题时可以查看源代码实现
