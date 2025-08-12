from abc import ABC, abstractmethod
from typing import Dict, Any, List
import copy
"""
ABC (Abstract Base Class) 是 Python 中用于创建抽象基类的工具
它的主要作用包括：
    通过 @abstractmethod 装饰器标记的方法必须在子类中实现，否则无法实例化该子类
    提供一个统一的接口规范，确保所有子类都实现了特定的方法
"""
class BaseClient(ABC):
    """客户端基类"""
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.model = None
        self.data = None
    
    @abstractmethod
    def train(self, global_model_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        训练本地模型
        
        Args:
            global_model_params: 全局模型参数
            
        Returns:
            本地训练后的模型参数
        """
        pass
    
    @abstractmethod
    def set_data(self, data):
        """设置客户端数据"""
        pass


class BaseServer(ABC):
    """服务器基类"""
    
    def __init__(self):
        self.global_model = None
        self.clients = []
    
    @abstractmethod
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        聚合客户端更新
        
        Args:
            client_updates: 客户端模型更新列表
            
        Returns:
            聚合后的全局模型参数
        """
        pass
    
    @abstractmethod
    def initialize_model(self):
        """初始化全局模型"""
        pass


class BaseModel(ABC):
    """模型基类"""
    
    def __init__(self, optimizer_config: Dict[str, Any] = None):
        """
        初始化模型基类
        
        Args:
            optimizer_config: 优化器配置
        """
        self.optimizer_config = optimizer_config
        self.optimizer = None
        
    def create_optimizer(self, model_parameters):
        """
        创建优化器 - 可被子类重写以支持自定义优化器
        
        Args:
            model_parameters: 模型参数
        """
        if self.optimizer_config:
            from utils.optimizer_factory import OptimizerFactory
            self.optimizer = OptimizerFactory.create_optimizer(
                model_parameters, self.optimizer_config
            )
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """获取模型参数"""
        pass
    
    @abstractmethod
    def set_parameters(self, params: Dict[str, Any]):
        """设置模型参数"""
        pass
    
    @abstractmethod
    def train_step(self, data, labels):
        """单步训练"""
        pass
    
    @abstractmethod
    def evaluate(self, data, labels):
        """模型评估"""
        pass


class BaseCommunication(ABC):
    """通信基类"""
    
    @abstractmethod
    def send_to_server(self, client_id: str, data: Any):
        """发送数据到服务器"""
        pass
    
    @abstractmethod
    def send_to_client(self, client_id: str, data: Any):
        """发送数据到客户端"""
        pass
    
    @abstractmethod
    def broadcast_to_clients(self, data: Any):
        """广播数据到所有客户端"""
        pass
