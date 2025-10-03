import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm
from data.templates import get_templates
from data.classnames import classnames_dict


class ClassificationHead(torch.nn.Linear):
    """
    分类头类
    继承自torch.nn.Linear，用于将特征向量映射到类别概率
    支持特征归一化功能
    """

    def __init__(self, normalize, weights, biases=None):
        """
        初始化分类头
        
        Args:
            normalize: 是否对输入特征进行L2归一化
            weights: 权重矩阵，形状为(output_size, input_size)
            biases: 偏置向量，可选参数
        """
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())

    def forward(self, inputs):
        # 如果需要归一化，对输入进行L2归一化
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)


def build_classification_head(
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        dataset_name: str,
        device: torch.device,
        logit_scale: float = 4.6052,
        head_dir: str = None
) -> ClassificationHead:
    """
    构建零样本分类头

    Args:
        tokenizer (CLIPTokenizer): 文本分词器
        text_encoder (CLIPTextModel): 文本编码器
        dataset_name (str): 数据集名称
        device (torch.device): 设备
        logit_scale (float): logit缩放因子（默认4.6052，对应exp(4.6052)≈100）
        head_dir (str): 分类头保存目录
    Returns:
        ClassificationHead: 分类头对象
    """
    os.makedirs(head_dir, exist_ok=True)
    head_path = os.path.join(head_dir, f"{dataset_name}_head.pth")
    # 如果缓存存在，直接加载
    if os.path.exists(head_path):
        print(f"Loading classification head from cache: {head_path}")
        weights = torch.load(head_path, map_location=device)
        classification_head = ClassificationHead(
            normalize=True,
            weights=weights
        )
        return classification_head
    
    text_encoder.eval()
    text_encoder.to(device)

    templates = get_templates(dataset_name.upper())
    classnames = classnames_dict[dataset_name.upper()]
    logit_scale_tensor = torch.tensor(logit_scale).to(device)

    print(f'Building classification head for dataset: {dataset_name}')
    zeroshot_weights = []

    with torch.no_grad():
        for classname in tqdm(classnames, desc="Encoding class names"):
            texts = [t(classname) for t in templates]
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = text_encoder(inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # 句子表示
            embeddings = F.normalize(embeddings, dim=-1)
            class_embedding = embeddings.mean(dim=0, keepdim=True)
            class_embedding = F.normalize(class_embedding, dim=-1)
            zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.cat(zeroshot_weights, dim=0)  # shape: [num_classes, feature_dim]
        zeroshot_weights = zeroshot_weights * logit_scale_tensor.exp()
        zeroshot_weights = zeroshot_weights.float()

    # 保存权重到缓存
    torch.save(zeroshot_weights.cpu(), head_path)
    print(f"Classification head saved to cache: {head_path}")

    classification_head = ClassificationHead(
        normalize=True,
        weights=zeroshot_weights
    )

    return classification_head


def create_multi_dataset_zero_shot_classifier(
        tokenizer,
        text_encoder,
        dataset_names: List[str],
        device: torch.device,
        head_dir: str = None
):
    multi_dataset_zero_shot_classifier = nn.ModuleDict()
    for dataset_name in dataset_names:
        multi_dataset_zero_shot_classifier[dataset_name] \
            = build_classification_head(tokenizer, text_encoder, dataset_name, device, head_dir=head_dir)
    return multi_dataset_zero_shot_classifier


class MultiHeadImageClassifier(torch.nn.Module):
    """
    多头图像分类器类
    支持多个分类头的图像分类器，可以根据头索引选择不同的分类头
    适用于多任务学习场景
    """

    def __init__(self, image_encoder, classification_heads):
        """
        初始化多头图像分类器
        
        Args:
            image_encoder: 图像编码器实例
            classification_heads: 分类头字典，每个头对应一个分类任务
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_heads = classification_heads
        self.is_lora_applied = False    # 用于标记LoRA是否应用

    def freeze_head(self):
        """
        冻结所有分类头的参数，使其在训练时不更新
        """
        for head in self.classification_heads.values():
            head.weight.requires_grad_(False)
            head.bias.requires_grad_(False)

    def forward(self, inputs, dataset_name="cifar100"):
        """
        前向传播
        
        Args:
            inputs: 输入图像
            dataset_name: 数据集名称
            
        Returns:
            对应分类头的分类结果
        """
        features = self.image_encoder(inputs)
        outputs = self.classification_heads[dataset_name](features)
        return outputs
