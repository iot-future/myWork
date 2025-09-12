"""
结果处理工具模块 (已优化)
负责实验结果的格式化和展示 - 现在主要由 EvaluationManager 处理
"""

from typing import Dict, Any, List


class ResultsHandler:
    """结果处理器 - 保留向后兼容性"""
    
    @staticmethod
    def print_experiment_summary(results: List[Dict[str, Any]]):
        """
        打印实验结果摘要 (已弃用)
        建议使用 EvaluationManager.get_final_summary() 代替
        """
        print("⚠️  建议使用 EvaluationManager 进行结果处理")
        if not results:
            print("⚠️  没有可用的实验结果")
            return
        
        # 简化版输出
        print(f"训练轮次: {len(results)}")
        
        if results:
            final_metrics = results[-1]
            accuracy = final_metrics.get('avg_accuracy', 0)
            loss = final_metrics.get('avg_loss', 0)
            
            # 转换准确率为百分比
            accuracy_display = accuracy * 100 if accuracy <= 1.0 else accuracy
            print(f"最终结果: 准确率 {accuracy_display:.2f}%, 损失 {loss:.4f}")
        
        print("✅ 实验完成")
    
    @staticmethod
    def format_training_progress(round_num: int, metrics: Dict[str, float]) -> str:
        """
        格式化训练进度输出 (已弃用)
        建议使用 EvaluationManager.format_round_summary() 代替
        """
        # 简化版格式化
        accuracy = metrics.get('avg_accuracy', 0)
        loss = metrics.get('avg_loss', 0)
        
        if accuracy == 0 and loss == 0:
            for key, value in metrics.items():
                if 'accuracy' in key and accuracy == 0:
                    accuracy = value
                elif 'loss' in key and loss == 0:
                    loss = value
        
        accuracy_display = accuracy * 100 if accuracy <= 1.0 else accuracy
        return f"轮次 {round_num:2d}: 准确率 {accuracy_display:.2f}%, 损失 {loss:.4f}"
