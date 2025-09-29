import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset


class SUN397(Dataset):
    """
    对 torchvision SUN397 数据集的封装。
    处理数据的下载、转换和访问。
    SUN397 包含397个类别的场景图像。
    """

    def __init__(self, data_root: str, train: bool = True, preprocess=None):
        """
        初始化 SUN397 数据集。

        Args:
            data_root (str): 存储或下载数据集的根目录。
            train (bool): 为 True 则从训练集创建数据集，否则从测试集创建。
            preprocess: 可选的预处理函数，如果提供则会覆盖默认的transform。
        """
        train_dir = os.path.join(data_root, 'sun397', 'train')
        val_dir = os.path.join(data_root, 'sun397', 'val')

        # 设置数据变换
        if preprocess is not None:
            transform = preprocess
        else:
            # SUN397 标准预处理：归一化到 [0,1] 然后使用 ImageNet 的均值和标准差
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        if train:
            self.dataset = datasets.ImageFolder(train_dir, transform=transform)
        else:
            self.dataset = datasets.ImageFolder(val_dir, transform=transform)

        # SUN397 的397个类别（这里只列出部分作为示例）
        self.classnames = ['abbey', 'airport_terminal', 'alley', 'amphitheater', 'apartment_building/outdoor',
                           'aquarium', 'arch', 'art_gallery', 'art_school', 'auditorium',
                           # ... (省略其他类别)
                           'youth_hostel', 'zen_garden']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
