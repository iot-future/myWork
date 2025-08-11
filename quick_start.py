"""
快速开始示例

这个示例展示如何快速使用联邦学习框架。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 使用框架的便捷函数
from __init__ import create_simple_fl_setup
from utils.logger import default_logger


def quick_start_example():
    """快速开始示例"""
    
    default_logger.info("=== 快速开始联邦学习示例 ===")
    
    # 1. 快速创建联邦学习设置
    fl_setup = create_simple_fl_setup(
        num_clients=3,
        n_features=10,
        n_classes=2,
        n_samples=600,
        learning_rate=0.1,
        batch_size=16,
        data_type="classification",
        iid=True
    )
    
    server = fl_setup["server"]
    clients = fl_setup["clients"]
    communication = fl_setup["communication"]
    test_data, test_labels = fl_setup["test_data"]
    
    default_logger.info(f"创建了 {len(clients)} 个客户端")
    default_logger.info(f"数据信息: {fl_setup['data_info']}")
    
    # 2. 运行联邦学习
    num_rounds = 5
    
    for round_num in range(1, num_rounds + 1):
        default_logger.log_round_start(round_num)
        
        # 获取全局模型参数
        global_params = server.send_global_model()
        
        # 广播到客户端
        communication.broadcast_to_clients(global_params)
        
        # 客户端训练
        client_updates = []
        for client in clients:
            # 接收全局参数
            messages = communication.receive_from_server(client.client_id)
            latest_params = messages[-1] if messages else global_params
            
            # 本地训练
            updated_params = client.train(latest_params)
            client_updates.append(updated_params)
            
            # 发送更新
            communication.send_to_server(client.client_id, updated_params)
        
        # 服务器聚合
        server.aggregate(client_updates)
        
        # 评估
        eval_results = server.evaluate_global_model(test_data, test_labels)
        default_logger.log_round_end(round_num, eval_results)
        
        # 清空缓冲区
        communication.clear_buffers()
    
    # 最终评估
    final_results = server.evaluate_global_model(test_data, test_labels)
    default_logger.log_evaluation(final_results, "final")
    
    default_logger.info("=== 快速开始示例完成 ===")
    
    return final_results


if __name__ == "__main__":
    results = quick_start_example()
    print(f"\n🎉 最终模型性能: {results}")
    print("\n✅ 联邦学习框架运行成功！")
