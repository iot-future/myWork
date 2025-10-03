import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import (
    CLIPModel,
    CLIPTokenizer
)
from peft import LoraConfig, get_peft_model
import os
import torch.optim as optim
from models.classificationHead import ClassificationHead
from tqdm import tqdm
import numpy as np
seed = 42
head_path = "/mnt/nfs/dataset/zzm_checkpoints/zero_shot_heads/cifar100_head.pth"
device = "cuda"
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = str(42)
torch.use_deterministic_algorithms(True)

class CIFAR100_CLIP_Zeroshot(torch.nn.Module):
    def __init__(self,
                 model_name="openai/clip-vit-base-patch32",
                 cache_dir="/mnt/nfs/dataset/zzm_checkpoints",
                 device="cuda"):
        super().__init__()
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir).to(device)
        del self.model.text_model
        del self.model.text_embed_dim
        del self.model.text_projection
        # 构建CIFAR-100的零样本分类头
        self.classification_head = ClassificationHead(normalize=True,
                                                      weights=torch.load(head_path, map_location=device)).to(device)
        self.classification_head.eval()  # 冻结零样本头

    def forward(self, pixel_values):
        # pixel_values: [B, 3, H, W]，已归一化
        features = self.model.get_image_features(pixel_values=pixel_values)
        logits = self.classification_head(features)
        return logits

    @torch.no_grad()
    def predict(self, pixel_values):
        logits = self.forward(pixel_values)
        return logits.argmax(dim=-1)  # 返回预测类别

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
# 加载CIFAR-100数据集
dataset_train = datasets.CIFAR100("/mnt/nfs/dataset", train=True, transform=transform)
dataset_test = datasets.CIFAR100("/mnt/nfs/dataset", train=False, transform=transform)

# 创建数据加载器
train_dataloader = DataLoader(
    dataset_train,
    shuffle=False,
    batch_size=32,
    num_workers=32,
    pin_memory=True
)
eval_dataloader = DataLoader(
    dataset_test,
    batch_size=32,
    num_workers=32,
    pin_memory=True
)

# 加载CLIP模型
model = CIFAR100_CLIP_Zeroshot(
    model_name="openai/clip-vit-base-patch32",
    cache_dir="/mnt/nfs/dataset/zzm_checkpoints",
    device=device
)
# 保存分类头的bias
torch.save(model.classification_head.bias, "/mnt/nfs/dataset/zzm_checkpoints/zero_shot_heads/cifar100_head_bias.pth")
print("Classification head bias saved.")
# 配置LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "out_proj",
        "visual.q_proj", "visual.k_proj", "visual.v_proj", "visual.out_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="FEATURE_EXTRACTION",
)
model.model = get_peft_model(model.model, lora_config)
model.model.print_trainable_parameters()

optimizer = optim.AdamW(
    model.parameters(),
    lr=5e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01
)

num_epochs = 10
# 打印模型参数

model.classification_head.weight.requires_grad_(False)
model.classification_head.bias.requires_grad_(False)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
    for batch in pbar:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=loss.item(), acc=correct / total)

    avg_loss = total_loss / total
    train_acc = correct / total
    print(f"Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")

    # 验证
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in eval_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total
    print(f"Val Acc: {val_acc:.4f}")
