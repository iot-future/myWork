#!/usr/bin/env python3
"""
配置文件验证脚本
用于检查联邦学习配置文件的正确性和一致性
"""

import yaml
import os
import sys
from pathlib import Path

def validate_config(config_path):
    """验证单个配置文件"""
    print(f"\n验证配置文件: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ YAML解析错误: {e}")
        return False
    
    errors = []
    warnings = []
    
    # 检查必需的顶级键
    required_keys = ['experiment', 'client', 'server', 'model', 'optimizer', 'data', 'evaluation']
    for key in required_keys:
        if key not in config:
            errors.append(f"缺少必需的顶级键: {key}")
    
    # 检查实验配置
    if 'experiment' in config:
        exp = config['experiment']
        if 'name' not in exp:
            errors.append("experiment.name 缺失")
        if 'rounds' not in exp or not isinstance(exp['rounds'], int) or exp['rounds'] <= 0:
            errors.append("experiment.rounds 必须是正整数")
    
    # 检查客户端配置
    if 'client' in config:
        client = config['client']
        if 'num_clients' not in client or not isinstance(client['num_clients'], int) or client['num_clients'] <= 0:
            errors.append("client.num_clients 必须是正整数")
        
        if 'client_datasets' in client:
            num_clients = client.get('num_clients', 0)
            client_datasets = client['client_datasets']
            
            # 检查客户端数量和数据集配置是否匹配
            expected_clients = [f"client_{i}" for i in range(num_clients)]
            actual_clients = list(client_datasets.keys())
            
            if len(actual_clients) != num_clients:
                warnings.append(f"客户端数量({num_clients})与client_datasets中的配置数量({len(actual_clients)})不匹配")
            
            for i in range(num_clients):
                client_key = f"client_{i}"
                if client_key not in client_datasets:
                    errors.append(f"缺少客户端配置: {client_key}")
    
    # 检查学习率一致性
    learning_rates = {}
    if 'client' in config and 'learning_rate' in config['client']:
        learning_rates['client'] = config['client']['learning_rate']
    if 'model' in config and 'learning_rate' in config['model']:
        learning_rates['model'] = config['model']['learning_rate']
    if 'optimizer' in config and 'learning_rate' in config['optimizer']:
        learning_rates['optimizer'] = config['optimizer']['learning_rate']
    
    if len(set(learning_rates.values())) > 1:
        warnings.append(f"学习率不一致: {learning_rates}")
    
    # 检查数据配置
    if 'data' in config:
        data = config['data']
        if 'data_dir' not in data:
            warnings.append("data.data_dir 未配置")
        if 'batch_size' not in data or not isinstance(data['batch_size'], int) or data['batch_size'] <= 0:
            errors.append("data.batch_size 必须是正整数")
    
    # 检查模型类型特定配置
    if 'model' in config and 'type' in config['model']:
        model_type = config['model']['type']
        if model_type == 'clip':
            # CLIP模型特有检查
            if 'model_name' not in config['model']:
                warnings.append("CLIP模型缺少model_name配置")
            if 'cache_dir' not in config['model']:
                warnings.append("CLIP模型缺少cache_dir配置")
    
    # 输出结果
    if errors:
        print("❌ 发现错误:")
        for error in errors:
            print(f"   - {error}")
    
    if warnings:
        print("⚠️  发现警告:")
        for warning in warnings:
            print(f"   - {warning}")
    
    if not errors and not warnings:
        print("✅ 配置文件验证通过")
    
    return len(errors) == 0

def main():
    """主函数"""
    configs_dir = Path(__file__).parent
    
    # 获取所有yaml配置文件
    config_files = []
    for file in configs_dir.glob("*.yaml"):
        if file.name not in ['template.yaml']:  # 跳过模板文件
            config_files.append(file)
    
    if not config_files:
        print("未找到配置文件")
        return 1
    
    print("=" * 60)
    print("联邦学习配置文件验证")
    print("=" * 60)
    
    all_valid = True
    for config_file in sorted(config_files):
        if not validate_config(config_file):
            all_valid = False
    
    print("\n" + "=" * 60)
    if all_valid:
        print("✅ 所有配置文件验证通过")
        return 0
    else:
        print("❌ 部分配置文件存在问题，请检查上面的错误信息")
        return 1

if __name__ == "__main__":
    sys.exit(main())
