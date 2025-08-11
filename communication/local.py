from typing import Any, Dict, List
from core.base import BaseCommunication
import copy


class LocalCommunication(BaseCommunication):
    """本地通信实现（用于单机模拟）"""
    
    def __init__(self):
        self.server_buffer = {}  # 服务器缓冲区
        self.client_buffers = {}  # 客户端缓冲区
        self.client_list = []  # 客户端列表
    
    def register_client(self, client_id: str):
        """注册客户端"""
        if client_id not in self.client_list:
            self.client_list.append(client_id)
            self.client_buffers[client_id] = []
    
    def send_to_server(self, client_id: str, data: Any):
        """发送数据到服务器"""
        if client_id not in self.server_buffer:
            self.server_buffer[client_id] = []
        
        # 深拷贝数据以避免引用问题
        self.server_buffer[client_id].append(copy.deepcopy(data))
    
    def send_to_client(self, client_id: str, data: Any):
        """发送数据到指定客户端"""
        if client_id not in self.client_buffers:
            self.client_buffers[client_id] = []
        
        # 深拷贝数据以避免引用问题
        self.client_buffers[client_id].append(copy.deepcopy(data))
    
    def broadcast_to_clients(self, data: Any):
        """广播数据到所有客户端"""
        for client_id in self.client_list:
            self.send_to_client(client_id, data)
    
    def receive_from_server(self, client_id: str) -> List[Any]:
        """客户端接收来自服务器的数据"""
        if client_id in self.client_buffers:
            messages = self.client_buffers[client_id].copy()
            self.client_buffers[client_id].clear()  # 清空缓冲区
            return messages
        return []
    
    def receive_from_clients(self) -> Dict[str, List[Any]]:
        """服务器接收来自所有客户端的数据"""
        messages = copy.deepcopy(self.server_buffer)
        self.server_buffer.clear()  # 清空缓冲区
        return messages
    
    def get_client_count(self) -> int:
        """获取注册的客户端数量"""
        return len(self.client_list)
    
    def clear_buffers(self):
        """清空所有缓冲区"""
        self.server_buffer.clear()
        for client_id in self.client_buffers:
            self.client_buffers[client_id].clear()
