"""
结果处理工具模块
负责实验结果的格式化和展示
"""

from typing import Dict, Any, List


class ResultsHandler:
    """结果处理器"""
    
    @staticmethod
    def print_experiment_summary(results: List[Dict[str, Any]]):
        """打印实验结果摘要"""
        if not results:
            print("⚠️  没有可用的实验结果")
            return
        
        # 打印训练过程
        print(f"训练轮次: {len(results)}")
        
        if results:
            # 打印最终结果
            final_metrics = results[-1]
            accuracy = final_metrics.get('accuracy', 0)
            loss = final_metrics.get('loss', 0)
            print(f"最终准确率: {accuracy:.4f}% | 最终损失: {loss:.4f}")
            
            # 打印训练趋势
            if len(results) > 1:
                first_metrics = results[0]
                accuracy_improvement = accuracy - first_metrics.get('accuracy', 0)
                loss_reduction = first_metrics.get('loss', 0) - loss
                print(f"准确率提升: +{accuracy_improvement:.4f}% | 损失降低: -{loss_reduction:.4f}")
        
        print("-" * 60)
        print("✅ 实验完成")
    
    @staticmethod
    def format_training_progress(round_num: int, metrics: Dict[str, float]) -> str:
        """格式化训练进度输出"""
        accuracy = metrics.get('accuracy', 0)
        loss = metrics.get('loss', 0)
        return f"轮次 {round_num:2d}: 准确率 {accuracy:.4f}% | 损失 {loss:.4f}"
