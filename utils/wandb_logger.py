"""
简洁的 WandB 记录器
记录客户端训练情况和全局模型表现
"""

import os
import wandb
from typing import Dict, Any, Optional


def init_wandb(config: Dict[str, Any], project_name: str = "federated-learning", offline: bool = False):
    """
    初始化 WandB
    
    Args:
        config: 实验配置
        project_name: 项目名称
        offline: 是否使用离线模式
    """
    if offline:
        os.environ["WANDB_MODE"] = "offline"
    else:
        os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
        os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    
    experiment_name = config.get('experiment', {}).get('name', 'fl_experiment')
    
    wandb.init(
        project=project_name,
        name=experiment_name,
        config=config
    )
    if offline:
        print("✓ WandB 离线模式已启用")
    else:
        print("✓ WandB 在线模式已启用")


def log_client_metrics(client_id: str, round_num: int, loss: float, accuracy: Optional[float] = None):
    """记录客户端训练指标"""
    metrics = {
        f"client/{client_id}/loss": loss,
        "round": round_num
    }
    if accuracy is not None:
        metrics[f"client/{client_id}/accuracy"] = accuracy
    
    wandb.log(metrics)


def log_global_metrics(round_num: int, metrics: Dict[str, float]):
    """记录全局模型指标"""
    log_data = {"round": round_num}
    for key, value in metrics.items():
        log_data[f"global/{key}"] = value
    
    wandb.log(log_data)


def finish_wandb():
    """结束 WandB 记录"""
    wandb.finish()
    print("✓ WandB 记录会话结束")