from torch.utils.data import Dataset
from torchvision import datasets, transforms


class EuroSAT(Dataset):
    """
    对 torchvision EuroSAT 数据集的封装。
    处理数据的下载、转换和访问。
    EuroSAT 包含10个类别的卫星图像。
    """
    
    def __init__(self, data_root: str, train: bool = True, preprocess=None):
        """
        初始化 EuroSAT 数据集。
        Args:
            data_root (str): 存储或下载数据集的根目录。
            train (bool): 为 True 则从训练集创建数据集，否则从测试集创建。
            preprocess: 可选的预处理函数，如果提供则会覆盖默认的transform。
        """
        # 设置数据变换
        if preprocess is not None:
            transform = preprocess
        else:
            # EuroSAT 标准预处理：归一化到 [0,1] 然后使用 ImageNet 的均值和标准差
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        self.dataset = datasets.EuroSAT(
            root=data_root,
            download=True,
            transform=transform
        )
        # EuroSAT 的10个类别
        self.classnames = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
                           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]