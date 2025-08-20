"""
定义了此项目中使用的所有数据集类。
每个数据集类都封装了从加载、转换到访问单个数据项的所有逻辑。
这种设计使得在联邦学习设置中添加新数据集变得简单。
"""
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class CIFAR10(Dataset):
    """
    对 torchvision CIFAR-10 数据集的封装。

    处理数据的下载、转换和访问。
    CIFAR-10 包含10个类别的32x32彩色图像。
    """

    def __init__(self, data_root: str, train: bool = True, preprocess=None):
        """
        初始化 CIFAR-10 数据集。

        Args:
            data_root (str): 存储或下载数据集的根目录。
            train (bool): 为 True 则从训练集创建数据集，否则从测试集创建。
            preprocess: 可选的预处理函数，如果提供则会覆盖默认的transform。
        """
        # 设置数据变换
        if preprocess is not None:
            transform = preprocess
        else:
            # CIFAR-10 标准预处理：归一化到 [0,1] 然后使用 CIFAR-10 的均值和标准差
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            
        self.dataset = datasets.CIFAR10(
            root=data_root,
            train=train,
            download=True,
            transform=transform
        )
        
        # CIFAR-10 的10个类别
        self.classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                          'dog', 'frog', 'horse', 'ship', 'truck']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
