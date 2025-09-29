import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Callable, Optional


class RESISC45(Dataset):
    """
    对 RESISC45 数据集的封装。

    处理数据的加载、转换和访问。
    RESISC45 包含 45 个类别的 256x256 彩色图像。
    """

    def __init__(self, data_root: str, train: bool = True, preprocess: Optional[Callable] = None):
        """
        初始化 RESISC45 数据集。

        Args:
            data_root (str): 数据集的根目录。
            train (bool): 为 True 则从训练集创建数据集，否则从测试集创建。
            preprocess: 可选的预处理函数，如果提供则会覆盖默认的 transform。
        """
        # 设置数据分割
        split = "train" if train else "test"

        # 设置数据变换
        if preprocess is not None:
            self.transform = preprocess
        else:
            # RESISC45 标准预处理：归一化到 [0,1]
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        # 加载数据集文件列表
        self.data_root = data_root
        self.split = split
        self.image_paths = []
        self.labels = []

        split_file = os.path.join(data_root, "resisc45", f"resisc45-{split}.txt")
        with open(os.path.join(split_file, "resisc45", f"resisc45-{split}.txt"), "r") as f:
            for line in f:
                path = line.strip()  # 去掉每行的首尾空格或换行符
                label = path.split("_")[0]  # 类别名是文件名中下划线前的部分
                self.image_paths.append(os.path.join(self.root, "resisc45", "NWPU-RESISC45", path))
                self.labels.append(RESISC45.classnames.index(label))  # 将类别名转换为类别索引

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        获取数据集中的一个样本。

        Args:
            idx: 样本索引。

        Returns:
            tuple: (图像, 标签)。
        """
        from PIL import Image

        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # 加载图像
        image = Image.open(image_path).convert("RGB")

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return image, label

    # RESISC45 的 45 个类别
    classnames = [
        "airplane", "airport", "baseball_diamond", "basketball_court", "beach",
        "bridge", "chaparral", "church", "circular_farmland", "cloud",
        "commercial_area", "dense_residential", "desert", "forest", "freeway",
        "golf_course", "ground_track_field", "harbor", "industrial_area", "intersection",
        "island", "lake", "meadow", "medium_residential", "mobile_home_park",
        "mountain", "overpass", "palace", "parking_lot", "railway",
        "railway_station", "rectangular_farmland", "river", "roundabout", "runway",
        "sea_ice", "ship", "snowberg", "sparse_residential", "stadium",
        "storage_tank", "tennis_court", "terrace", "thermal_power_station", "wetland"
    ]
