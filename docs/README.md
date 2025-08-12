# 联邦学习框架文档

欢迎使用联邦学习框架！本文档系统将帮助您从零开始学习和使用我们的联邦学习平台。

## 📚 文档结构

### [01 - 入门指南](./01-getting-started/)
新手必读，快速上手联邦学习框架
- [快速开始](./01-getting-started/quick_start.md) - 5分钟快速体验联邦学习

### [02 - 架构设计](./02-architecture/)
深入了解框架的设计理念和架构
- [系统架构](./02-architecture/architecture.md) - 框架整体架构设计

### [03 - 使用教程](./03-tutorials/)
详细的使用教程和实践指南
- [实验运行指南](./03-tutorials/experiment_guide.md) - 如何配置和运行实验
- [数据使用指南](./03-tutorials/data_guide.md) - 数据加载和处理
- [CLIP模型指南](./03-tutorials/clip_guide.md) - CLIP模型集成使用

### [04 - 开发指南](./04-development/)
面向开发者的扩展和定制指南
- [新增模型指南](./04-development/how_to_add_new_model.md) - 如何添加新的机器学习模型
- [优化器指南](./04-development/optimizer_guide.md) - 优化器配置和扩展
- [数据变换最佳实践](./04-development/data_transform_best_practices.md) - 数据预处理最佳实践
- [WandB集成指南](./04-development/wandb_guide.md) - 实验追踪和可视化
- [重构指南](./04-development/refactoring_guide.md) - 代码重构和维护
- [扩展开发](./04-development/extensions/) - 框架扩展开发指南

### [05 - API参考](./05-api-reference/)
完整的API文档和参考手册
- [核心模块](./05-api-reference/modules/) - 核心组件API文档
  - [客户端API](./05-api-reference/modules/client.md)
  - [服务器API](./05-api-reference/modules/server.md)
- [模型参考](./05-api-reference/quick_model_reference.md) - 所有可用模型的快速参考

## 🚀 学习路径建议

### 初学者路径
1. 📖 阅读 [快速开始](./01-getting-started/quick_start.md)
2. 🏗️ 了解 [系统架构](./02-architecture/architecture.md)
3. 🧪 跟随 [实验运行指南](./03-tutorials/experiment_guide.md)
4. 📊 学习 [数据使用指南](./03-tutorials/data_guide.md)

### 开发者路径
1. 🎯 完成初学者路径
2. 🔧 阅读 [新增模型指南](./04-development/how_to_add_new_model.md)
3. ⚙️ 学习 [优化器指南](./04-development/optimizer_guide.md)
4. 📈 掌握 [WandB集成指南](./04-development/wandb_guide.md)
5. 🔍 参考 [API文档](./05-api-reference/)

### 高级用户路径
1. 📝 阅读 [重构指南](./04-development/refactoring_guide.md)
2. 🧩 探索 [扩展开发](./04-development/extensions/)
3. 💡 学习 [数据变换最佳实践](./04-development/data_transform_best_practices.md)

## 📋 快速参考

- **模型类型**: 线性模型、CNN、CLIP
- **数据集**: MNIST、自定义CSV数据
- **聚合算法**: FedAvg
- **优化器**: Adam、AdamW、SGD
- **实验追踪**: WandB集成
- **通信方式**: 本地模拟

## 🆘 获取帮助

- 查看具体模块的文档以获得详细信息
- 每个目录下都有相应的README文件
- 遇到问题时请先查看相关教程和API文档

## 📝 文档维护

本文档系统按功能模块分类，便于维护和查找：
- **01-getting-started**: 入门文档，帮助新用户快速上手
- **02-architecture**: 架构文档，理解系统设计
- **03-tutorials**: 教程文档，实践指导
- **04-development**: 开发文档，扩展和定制
- **05-api-reference**: API文档，技术参考

每个类别都有明确的目标用户和使用场景，确保文档的实用性和可维护性。
