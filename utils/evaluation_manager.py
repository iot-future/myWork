"""
统一评估管理器
负责联邦学习中客户端和服务器的评估逻辑
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """评估结果数据类"""
    round_num: int
    client_metrics: Dict[str, Dict[str, float]]  # {client_id: {metric_name: value}}
    global_metrics: Dict[str, float]  # {metric_name: value}
    dataset_sample_counts: Optional[Dict[str, int]] = None


class EvaluationManager:
    """
    统一评估管理器
    
    负责：
    1. 客户端本地评估
    2. 全局模型评估  
    3. 评估结果格式化
    4. 评估输出管理
    """
    
    def __init__(self, verbose: bool = True):
        """
        初始化评估管理器
        
        Args:
            verbose: 是否输出详细信息
        """
        self.verbose = verbose
        self.evaluation_history: List[EvaluationResult] = []
    
    def evaluate_clients(self, clients: List, round_num: int) -> Dict[str, Dict[str, float]]:
        """
        评估所有客户端的本地模型
        
        Args:
            clients: 客户端列表
            round_num: 当前轮次
            
        Returns:
            客户端评估结果字典
        """
        client_metrics = {}
        
        if self.verbose:
            print(f"  📱 评估客户端本地模型...")
            
        for client in clients:
            try:
                metrics = client.evaluate_on_local_data()
                if metrics:
                    client_metrics[client.client_id] = metrics
                    if self.verbose:
                        acc = metrics.get('accuracy', 0) * 100  # 转换为百分比
                        loss = metrics.get('loss', 0)
                        print(f"    {client.client_id}: 准确率 {acc:.2f}%, 损失 {loss:.4f}")
                        
            except Exception as e:
                if self.verbose:
                    print(f"    ⚠️  {client.client_id} 评估失败: {str(e)}")
                    
        return client_metrics
    
    def evaluate_global_model(self, server, test_loaders: Dict, round_num: int) -> Dict[str, float]:
        """
        评估全局模型
        
        Args:
            server: 联邦学习服务器
            test_loaders: 测试数据加载器字典
            round_num: 当前轮次
            
        Returns:
            全局模型评估结果字典
        """
        if not test_loaders:
            if self.verbose:
                print("    ⚠️  无测试数据集，跳过全局评估")
            return {}
        
        if self.verbose:
            print(f"  🌐 评估全局模型...")
            
        all_metrics = {}
        dataset_sample_counts = {}
        failed_count = 0
        
        # 评估每个数据集
        for dataset_name, test_loader in test_loaders.items():
            try:
                dataset_metrics = server.evaluate_with_dataloader(test_loader)
                if dataset_metrics:
                    dataset_sample_counts[dataset_name] = len(test_loader.dataset)
                    
                    # 添加数据集前缀
                    for metric_name, value in dataset_metrics.items():
                        all_metrics[f"{dataset_name}_{metric_name}"] = value
                        
                    if self.verbose:
                        acc = dataset_metrics.get('accuracy', 0) * 100
                        loss = dataset_metrics.get('loss', 0)
                        print(f"    {dataset_name}: 准确率 {acc:.2f}%, 损失 {loss:.4f}")
                        
            except Exception as e:
                failed_count += 1
                if self.verbose:
                    print(f"    ⚠️  {dataset_name} 评估失败: {str(e)}")
        
        # 计算加权平均指标
        if len(dataset_sample_counts) > 0:
            avg_metrics = self._calculate_weighted_average(all_metrics, dataset_sample_counts)
            all_metrics.update(avg_metrics)
            
            if self.verbose and avg_metrics:
                avg_acc = avg_metrics.get('avg_accuracy', 0) * 100
                avg_loss = avg_metrics.get('avg_loss', 0)
                print(f"    📊 平均: 准确率 {avg_acc:.2f}%, 损失 {avg_loss:.4f}")
        
        if failed_count > 0 and self.verbose:
            print(f"    ⚠️  {failed_count} 个数据集评估失败")
            
        return all_metrics
    
    def _calculate_weighted_average(self, all_metrics: Dict[str, float], 
                                  dataset_sample_counts: Dict[str, int]) -> Dict[str, float]:
        """计算加权平均指标"""
        avg_metrics = {}
        
        # 提取指标名称
        metric_names = set()
        for key in all_metrics.keys():
            for dataset_name in dataset_sample_counts.keys():
                if key.startswith(f"{dataset_name}_"):
                    metric_name = key[len(f"{dataset_name}_"):]
                    metric_names.add(metric_name)
                    break
        
        # 计算加权平均
        total_samples = sum(dataset_sample_counts.values())
        
        for metric_name in metric_names:
            weighted_sum = 0.0
            valid_samples = 0
            
            for dataset_name, sample_count in dataset_sample_counts.items():
                metric_key = f"{dataset_name}_{metric_name}"
                if metric_key in all_metrics:
                    weighted_sum += all_metrics[metric_key] * sample_count
                    valid_samples += sample_count
            
            if valid_samples > 0:
                avg_metrics[f"avg_{metric_name}"] = weighted_sum / valid_samples
                
        return avg_metrics
    
    def create_evaluation_result(self, round_num: int, client_metrics: Dict[str, Dict[str, float]], 
                               global_metrics: Dict[str, float]) -> EvaluationResult:
        """创建评估结果对象"""
        result = EvaluationResult(
            round_num=round_num,
            client_metrics=client_metrics,
            global_metrics=global_metrics
        )
        self.evaluation_history.append(result)
        return result
    
    def format_round_summary(self, result: EvaluationResult) -> str:
        """格式化轮次总结"""
        round_num = result.round_num
        global_metrics = result.global_metrics
        
        # 优先使用平均指标
        accuracy = global_metrics.get('avg_accuracy', 0)
        loss = global_metrics.get('avg_loss', 0)
        
        # 如果没有平均指标，使用第一个可用的指标
        if accuracy == 0 and loss == 0:
            for key, value in global_metrics.items():
                if 'accuracy' in key and accuracy == 0:
                    accuracy = value
                elif 'loss' in key and loss == 0:
                    loss = value
        
        # 转换准确率为百分比
        accuracy_pct = accuracy * 100 if accuracy <= 1.0 else accuracy
        
        return f"轮次 {round_num:2d}: 全局准确率 {accuracy_pct:.2f}%, 全局损失 {loss:.4f}"
    
    def get_final_summary(self) -> Dict[str, Any]:
        """获取最终实验总结"""
        if not self.evaluation_history:
            return {}
            
        final_result = self.evaluation_history[-1]
        first_result = self.evaluation_history[0]
        
        # 获取最终和初始指标
        final_global = final_result.global_metrics
        first_global = first_result.global_metrics
        
        # 计算改进情况
        final_acc = final_global.get('avg_accuracy', 0)
        final_loss = final_global.get('avg_loss', 0)
        first_acc = first_global.get('avg_accuracy', 0)  
        first_loss = first_global.get('avg_loss', 0)
        
        # 转换为百分比
        final_acc_pct = final_acc * 100 if final_acc <= 1.0 else final_acc
        first_acc_pct = first_acc * 100 if first_acc <= 1.0 else first_acc
        
        return {
            'total_rounds': len(self.evaluation_history),
            'final_accuracy': final_acc_pct,
            'final_loss': final_loss,
            'accuracy_improvement': final_acc_pct - first_acc_pct,
            'loss_reduction': first_loss - final_loss,
            'client_count': len(final_result.client_metrics)
        }
