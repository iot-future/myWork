"""
定义了此项目中使用的所有数据集类。
每个数据集类都封装了从加载、转换到访问单个数据项的所有逻辑。
这种设计使得在联邦学习设置中添加新数据集变得简单。
"""
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MNIST(Dataset):
    """
    对 torchvision MNIST 数据集的封装。

    处理数据的下载、转换和访问。
    """

    def __init__(self, data_root: str, train: bool = True, preprocess=None):
        """
        初始化 MNIST 数据集。

        Args:
            data_root (str): 存储或下载数据集的根目录。
            train (bool): 为 True 则从训练集创建数据集，否则从测试集创建。
            preprocess: 可选的预处理函数，如果提供则会覆盖默认的transform。
        """
        # 设置数据变换
        if preprocess is not None:
            transform = preprocess
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
        self.dataset = datasets.MNIST(
            root=data_root,
            train=train,
            download=True,
            transform=transform
        )
        
        self.classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]