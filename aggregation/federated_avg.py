from typing import List, Dict, Any
import copy


class FederatedAveraging:
    """联邦平均算法实现"""
    
    def __init__(self):
        pass
    
    def aggregate(self, client_updates: List[Dict[str, Any]], client_weights: Dict[str, float] = None) -> Dict[str, Any]:
        """
        执行联邦平均聚合
        
        Args:
            client_updates: 客户端更新列表，每个元素包含 'parameters' 和 'metrics'
            client_weights: 客户端权重字典，如果为None则使用均等权重
            
        Returns:
            聚合后的模型参数
        """
        if not client_updates:
            raise ValueError("No client updates provided")
        
        # 提取客户端参数
        client_params_list = []
        for update in client_updates:
            if isinstance(update, dict) and 'parameters' in update:
                client_params_list.append(update['parameters'])
            else:
                # 向后兼容：如果直接是参数字典
                client_params_list.append(update)
        
        # 如果没有提供权重或权重为空，使用均等权重
        if client_weights is None or len(client_weights) == 0:
            num_clients = len(client_params_list)
            weights = [1.0 / num_clients] * num_clients
        else:
            # 使用提供的权重
            weights = list(client_weights.values())
            # 归一化权重
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
        
        # 初始化聚合参数
        aggregated_params = None
        
        for i, client_params in enumerate(client_params_list):
            weight = weights[i]
            
            if aggregated_params is None:
                # 第一个客户端，直接复制参数
                aggregated_params = {}
                for key, value in client_params.items():
                    if hasattr(value, 'clone'):  # PyTorch tensor
                        aggregated_params[key] = value.clone() * weight
                    else:  # numpy array
                        aggregated_params[key] = value.copy() * weight
            else:
                # 累加其他客户端的参数
                for key, value in client_params.items():
                    if hasattr(value, 'clone'):  # PyTorch tensor
                        aggregated_params[key] += value * weight
                    else:  # numpy array
                        aggregated_params[key] += value * weight
        
        return aggregated_params


# 为了向后兼容，创建别名
FedAvgAggregator = FederatedAveraging
