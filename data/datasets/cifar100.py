"""
CIFAR-100数据集类实现。

CIFAR-100是CIFAR-10的扩展版本，包含100个不同的类别。
每个类别包含600张32x32的彩色图像，总共60,000张图像。
这100个类别被分为20个超类，每个超类包含5个子类。
"""
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class CIFAR100(Dataset):
    """
    对 torchvision CIFAR-100 数据集的封装。

    处理数据的下载、转换和访问。
    CIFAR-100 包含100个类别的32x32彩色图像。
    """

    def __init__(self, data_root: str, train: bool = True, preprocess=None):
        """
        初始化 CIFAR-100 数据集。

        Args:
            data_root (str): 存储或下载数据集的根目录。
            train (bool): 为 True 则从训练集创建数据集，否则从测试集创建。
            preprocess: 可选的预处理函数，如果提供则会覆盖默认的transform。
        """
        # 设置数据变换
        if preprocess is not None:
            transform = preprocess
        else:
            # CIFAR-100 标准预处理：归一化到 [0,1] 然后使用 CIFAR-100 的均值和标准差
            # CIFAR-100 的均值和标准差与 CIFAR-10 相似但略有不同
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
            
        self.dataset = datasets.CIFAR100(
            root=data_root,
            train=train,
            download=True,
            transform=transform
        )
        
        # CIFAR-100 的100个类别名称
        self.classnames = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver',
            'bed', 'bee', 'beetle', 'bicycle', 'bottle',
            'bowl', 'boy', 'bridge', 'bus', 'butterfly',
            'camel', 'can', 'castle', 'caterpillar', 'cattle',
            'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',
            'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox',
            'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
            'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
            'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
            'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
            'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
            'plain', 'plate', 'poppy', 'porcupine', 'possum',
            'rabbit', 'raccoon', 'ray', 'road', 'rocket',
            'rose', 'sea', 'seal', 'shark', 'shrew',
            'skunk', 'skyscraper', 'snail', 'snake', 'spider',
            'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor',
            'train', 'trout', 'tulip', 'turtle', 'wardrobe',
            'whale', 'willow_tree', 'wolf', 'woman', 'worm'
        ]

        # CIFAR-100 的20个超类名称（可选，用于分层分类）
        self.coarse_classnames = [
            'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
            'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
            'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
            'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles',
            'small_mammals', 'trees', 'vehicles_1', 'vehicles_2'
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]