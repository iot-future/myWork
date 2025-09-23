from typing import List, Dict, Any
from .base import BaseServer
from copy import deepcopy

class FederatedServer(BaseServer):
    """联邦学习服务器实现"""

    def __init__(self, global_model, aggregator):
        super().__init__()
        self.global_model = global_model
        self.aggregator = aggregator
        self.round_num = 0
        self.client_weights = {}  # 客户端权重

    def initialize_model(self):
        """初始化全局模型"""
        # 模型已在构造函数中设置
        pass

    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        聚合客户端更新
        """
        if not client_updates:
            return self.global_model.get_parameters()

        # 使用聚合算法进行聚合
        aggregated_params = self.aggregator.aggregate(
            client_updates, self.client_weights
        )

        aggregated_params = deepcopy(aggregated_params)
        # 更新全局模型
        self.global_model.set_parameters(aggregated_params)
        self.round_num += 1

        return aggregated_params

    def send_global_model(self) -> Dict[str, Any]:
        """发送全局模型参数给客户端"""
        return self.global_model.get_parameters()
    
    def get_global_model_parameters(self) -> Dict[str, Any]:
        """获取全局模型参数"""
        return self.global_model.get_parameters()
    
    def set_client_weights(self, client_weights: Dict[str, float]):
        """设置客户端权重"""
        self.client_weights = client_weights

    def evaluate(self, test_dataloader):
        """使用数据加载器评估全局模型"""
        try:
            return self.global_model.evaluate(test_dataloader)
        except Exception as e:
            print(f"⚠️  全局模型数据加载器评估失败: {str(e)}")
            return {}
