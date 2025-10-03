#!/usr/bin/env python3
"""
CLIP + LoRA 训练 CIFAR-10 数据集 Demo

这个演示展示了如何使用项目中的CLIP模型和LoRA模块来训练CIFAR-10数据集。
配置：
- 数据集：CIFAR-10（25000个样本）
- LoRA参数：r=8, alpha=16
- 批大小：32
- 包含训练进度条
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import time
from typing import Dict, Any, Tuple

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 导入项目模块
from models.clip import ImageEncoder
from lora.loRA_utils import LoRAWrapper
from lora.lora_config import LoRAConfig
from data.datasets.cifar10 import CIFAR10


class CLIPClassifier(nn.Module):
    """CLIP图像编码器 + 分类头"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", num_classes: int = 10, cache_dir: str = None):
        super().__init__()
        self.image_encoder = ImageEncoder(model_name=model_name, cache_dir=cache_dir)
        # 获取CLIP图像特征维度（从vision_model的hidden_size获取）
        feature_dim = self.image_encoder.vision_model.config.hidden_size
        
        # 添加分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(feature_dim, num_classes)
        )
        
    def forward(self, pixel_values):
        # 获取图像特征
        image_features = self.image_encoder(pixel_values)
        # 分类
        logits = self.classifier(image_features)
        return logits


def create_cifar10_dataloader(data_root: str = "./data", 
                             batch_size: int = 32, 
                             max_samples: int = 25000,
                             train: bool = True) -> DataLoader:
    """创建CIFAR-10数据加载器"""
    
    # CLIP模型的预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # CLIP需要224x224输入
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP预训练的标准化参数
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    # 创建数据集
    dataset = CIFAR10(data_root=data_root, train=train, preprocess=transform)
    
    # 如果需要限制样本数量，创建子集
    if max_samples < len(dataset):
        indices = list(range(max_samples))
        dataset = Subset(dataset, indices)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=2,
        pin_memory=True
    )
    
    return dataloader


def apply_lora_to_model(model: CLIPClassifier, r: int = 8, alpha: int = 16) -> LoRAWrapper:
    """为CLIP模型应用LoRA"""
    
    print(f"正在应用LoRA配置: r={r}, alpha={alpha}")
    
    # 创建LoRA包装器
    lora_wrapper = LoRAWrapper(model.image_encoder.vision_model)
    
    # 应用LoRA配置
    lora_wrapper.apply_lora(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
    )
    
    return lora_wrapper


def train_epoch(model: CLIPClassifier, 
                dataloader: DataLoader, 
                optimizer: optim.Optimizer, 
                criterion: nn.Module, 
                device: torch.device,
                epoch: int) -> Tuple[float, float]:
    """训练一个epoch"""
    
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # 创建进度条
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 更新进度条
        current_acc = 100. * correct / total
        current_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({
            'Loss': f'{current_loss:.4f}',
            'Acc': f'{current_acc:.2f}%',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    epoch_loss = total_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model: CLIPClassifier, 
             dataloader: DataLoader, 
             criterion: nn.Module, 
             device: torch.device) -> Tuple[float, float]:
    """评估模型"""
    
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            current_acc = 100. * correct / total
            pbar.set_postfix({'Acc': f'{current_acc:.2f}%'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def main():
    """主训练函数"""
    
    print("🚀 CLIP + LoRA CIFAR-10 训练 Demo")
    print("=" * 50)
    
    # 配置参数
    config = {
        'batch_size': 32,
        'max_samples': 25000,
        'lora_r': 8,
        'lora_alpha': 16,
        'learning_rate': 2e-5,  # 降低学习率
        'epochs': 5,
        'data_root': '/home/zzm/dataset',  # CIFAR-10数据集路径
        'model_name': "openai/clip-vit-base-patch32",
        'cache_dir': '/mnt/nfs/dataset/zzm_checkpoints'  # CLIP模型缓存路径
    }
    
    # 创建必要的目录
    os.makedirs(config['data_root'], exist_ok=True)
    os.makedirs(config['cache_dir'], exist_ok=True)
    
    print("配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    print(f"数据集路径: {config['data_root']}")
    print(f"模型缓存路径: {config['cache_dir']}")
    print()
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print()
    
    # 创建数据加载器
    print("🔄 正在准备数据...")
    train_loader = create_cifar10_dataloader(
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        max_samples=config['max_samples'],
        train=True
    )
    
    test_loader = create_cifar10_dataloader(
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        max_samples=5000,  # 测试集使用5000个样本
        train=False
    )
    
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"测试样本数: {len(test_loader.dataset)}")
    print(f"批大小: {config['batch_size']}")
    print(f"训练批次数: {len(train_loader)}")
    print()
    
    # 创建模型
    print("🔧 正在初始化模型...")
    model = CLIPClassifier(
        model_name=config['model_name'], 
        num_classes=10,
        cache_dir=config['cache_dir']
    ).to(device)
    
    # 应用LoRA
    lora_wrapper = apply_lora_to_model(
        model, 
        r=config['lora_r'], 
        alpha=config['lora_alpha']
    )
    print()
    
    # 设置优化器和损失函数
    # 只训练LoRA参数和分类头
    lora_params = [p for p in model.image_encoder.vision_model.parameters() if p.requires_grad]
    classifier_params = list(model.classifier.parameters())
    trainable_params = lora_params + classifier_params
    
    optimizer = optim.AdamW(trainable_params, lr=config['learning_rate'], weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print(f"可训练参数数量: {sum(p.numel() for p in trainable_params):,}")
    print(f"总参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # 训练循环
    print("🎯 开始训练...")
    best_acc = 0.0
    
    for epoch in range(config['epochs']):
        print(f"\n📈 Epoch {epoch+1}/{config['epochs']}")
        
        # 训练
        start_time = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        train_time = time.time() - start_time
        
        # 评估
        start_time = time.time()
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        eval_time = time.time() - start_time
        
        # 打印结果
        print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% ({train_time:.1f}s)")
        print(f"测试 - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}% ({eval_time:.1f}s)")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            print(f"🎉 新的最佳准确率: {best_acc:.2f}%")
            
            # 保存LoRA权重
            torch.save({
                'lora_state_dict': {name: param for name, param in model.named_parameters() if param.requires_grad},
                'classifier_state_dict': model.classifier.state_dict(),
                'config': config,
                'best_acc': best_acc
            }, 'best_clip_lora_cifar10.pth')
    
    print(f"\n✅ 训练完成！最佳测试准确率: {best_acc:.2f}%")
    print("模型已保存到: best_clip_lora_cifar10.pth")


if __name__ == "__main__":
    main()