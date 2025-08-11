import logging
import sys
from datetime import datetime


class FederatedLogger:
    """联邦学习日志工具"""
    
    def __init__(self, name: str = "FederatedLearning", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # 避免重复添加处理器
        if not self.logger.handlers:
            # 创建控制台处理器
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            
            # 创建格式器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            
            # 添加处理器到日志器
            self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        """记录信息日志"""
        self.logger.info(message)
    
    def debug(self, message: str):
        """记录调试日志"""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """记录警告日志"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """记录错误日志"""
        self.logger.error(message)
    
    def log_round_start(self, round_num: int):
        """记录训练轮次开始"""
        self.info(f"开始第 {round_num} 轮联邦学习训练")
    
    def log_round_end(self, round_num: int, metrics: dict):
        """记录训练轮次结束"""
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"第 {round_num} 轮训练完成 - {metrics_str}")
    
    def log_client_training(self, client_id: str, metrics: dict = None):
        """记录客户端训练信息"""
        if metrics:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.info(f"客户端 {client_id} 训练完成 - {metrics_str}")
        else:
            self.info(f"客户端 {client_id} 开始训练")
    
    def log_aggregation(self, num_clients: int):
        """记录聚合信息"""
        self.info(f"开始聚合 {num_clients} 个客户端的模型参数")
    
    def log_evaluation(self, metrics: dict, data_type: str = "test"):
        """记录评估结果"""
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"{data_type.capitalize()} 评估结果: {metrics_str}")


# 创建默认日志器实例
default_logger = FederatedLogger()
