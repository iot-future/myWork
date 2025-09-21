#!/usr/bin/env python3
"""
联邦学习实验入口文件
"""

import yaml
from utils.config_manager import ConfigManager
from utils.experiment_runner import ExperimentRunner
from utils.results_handler import ResultsHandler


def main():
    """主函数 - 简洁的实验流程"""
    # 创建配置管理器
    config_manager = ConfigManager()
    
    # 解析命令行参数
    parser = config_manager.create_parser()
    args = parser.parse_args()
    
    # 加载配置并应用命令行覆盖
    config = config_manager.load_config(args.config)
    config = config_manager.override_config(config, args)
    
    # 打印关键配置信息
    print("=" * 60)
    print("🔧 实验配置")
    print("=" * 60)
    print(f"实验名称: {config['experiment']['name']}")
    print(f"训练轮次: {config['experiment']['rounds']} | 客户端数量: {config['client']['num_clients']} | 本地训练轮次: {config['client']['local_epochs']}")
    print(f"优化器: AdamW (统一使用)")
    
    # 显示LoRA配置信息
    model_config = config.get('model', {})
    lora_config = model_config.get('lora', {})
    if lora_config.get('enabled', False):
        print(f"🎯 LoRA微调: 启用 (r={lora_config.get('r', 16)}, alpha={lora_config.get('lora_alpha', 32)})")
    else:
        print(f"📸 训练模式: 标准微调")
    
    data_info = f"批大小: {config['data']['batch_size']}"
    if config.get('wandb', {}).get('enabled', False):
        data_info += f" | WandB项目: {config['wandb']['experiment_name']}"
    print(data_info)
    print("=" * 60)
    
    # 运行实验
    print("🚀 开始运行实验...")
    print("-" * 60)
    experiment_runner = ExperimentRunner(config)
    results = experiment_runner.run_experiment()
    
    # 打印结果摘要
    print()
    print("=" * 60)
    print("📊 实验结果摘要")
    print("=" * 60)
    ResultsHandler.print_experiment_summary(results)
    print("=" * 60)


if __name__ == '__main__':
    main()
