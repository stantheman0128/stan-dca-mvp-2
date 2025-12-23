# core/statistics.py
"""統計檢驗模組"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class TTestResult:
    """t 檢驗結果"""
    t_statistic: float
    p_value: float
    significant: bool
    conclusion: str
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float


@dataclass
class ConfidenceInterval:
    """信心區間"""
    mean: float
    lower: float
    upper: float
    confidence_level: float


class StatisticsCalculator:
    """
    統計檢驗計算器
    
    提供 t 檢驗、信心區間等統計分析功能
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        初始化統計計算器
        
        Args:
            significance_level: 顯著水平（預設 0.05）
        """
        self.significance_level = significance_level
    
    def compare_strategies_ttest(
        self,
        returns_a: pd.Series,
        returns_b: pd.Series,
        strategy_a_name: str = "策略 A",
        strategy_b_name: str = "策略 B"
    ) -> TTestResult:
        """
        比較兩個策略的報酬是否有顯著差異（獨立樣本 t 檢驗）
        
        Args:
            returns_a: 策略 A 的報酬序列
            returns_b: 策略 B 的報酬序列
            strategy_a_name: 策略 A 名稱
            strategy_b_name: 策略 B 名稱
            
        Returns:
            TTestResult 物件
        """
        # 清理數據
        a = returns_a.dropna().values
        b = returns_b.dropna().values
        
        if len(a) < 3 or len(b) < 3:
            return TTestResult(
                t_statistic=0.0,
                p_value=1.0,
                significant=False,
                conclusion="數據不足，無法進行統計檢驗",
                mean_a=np.mean(a) if len(a) > 0 else 0,
                mean_b=np.mean(b) if len(b) > 0 else 0,
                std_a=np.std(a) if len(a) > 0 else 0,
                std_b=np.std(b) if len(b) > 0 else 0
            )
        
        # 執行 t 檢驗
        t_stat, p_value = stats.ttest_ind(a, b)
        
        significant = p_value < self.significance_level
        
        # 生成結論
        mean_diff = np.mean(a) - np.mean(b)
        if significant:
            if mean_diff > 0:
                conclusion = f"{strategy_a_name} 顯著優於 {strategy_b_name} (p = {p_value:.4f})"
            else:
                conclusion = f"{strategy_b_name} 顯著優於 {strategy_a_name} (p = {p_value:.4f})"
        else:
            conclusion = f"兩策略差異不顯著 (p = {p_value:.4f})"
        
        return TTestResult(
            t_statistic=round(t_stat, 4),
            p_value=round(p_value, 4),
            significant=significant,
            conclusion=conclusion,
            mean_a=round(np.mean(a), 4),
            mean_b=round(np.mean(b), 4),
            std_a=round(np.std(a), 4),
            std_b=round(np.std(b), 4)
        )
    
    def calculate_confidence_interval(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> ConfidenceInterval:
        """
        計算報酬的信心區間
        
        Args:
            returns: 報酬序列
            confidence_level: 信心水準（預設 0.95）
            
        Returns:
            ConfidenceInterval 物件
        """
        data = returns.dropna().values
        
        if len(data) < 3:
            return ConfidenceInterval(
                mean=0.0,
                lower=0.0,
                upper=0.0,
                confidence_level=confidence_level
            )
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        n = len(data)
        
        # 使用 t 分佈計算信心區間
        t_critical = stats.t.ppf((1 + confidence_level) / 2, df=n-1)
        margin = t_critical * (std / np.sqrt(n))
        
        return ConfidenceInterval(
            mean=round(mean, 4),
            lower=round(mean - margin, 4),
            upper=round(mean + margin, 4),
            confidence_level=confidence_level
        )
    
    def multiple_comparison(
        self,
        strategy_returns: Dict[str, pd.Series],
        baseline: Optional[str] = None
    ) -> pd.DataFrame:
        """
        多策略兩兩比較（含 Bonferroni 校正）
        
        Args:
            strategy_returns: {策略名稱: 報酬序列} 字典
            baseline: 基準策略名稱（若指定，只比較其他策略與基準）
            
        Returns:
            比較結果 DataFrame
        """
        strategies = list(strategy_returns.keys())
        results = []
        
        if baseline and baseline in strategies:
            # 只與基準比較
            comparisons = [(baseline, s) for s in strategies if s != baseline]
        else:
            # 兩兩比較
            comparisons = [(strategies[i], strategies[j]) 
                          for i in range(len(strategies)) 
                          for j in range(i+1, len(strategies))]
        
        # Bonferroni 校正
        n_comparisons = len(comparisons)
        adjusted_alpha = self.significance_level / n_comparisons if n_comparisons > 0 else self.significance_level
        
        for name_a, name_b in comparisons:
            result = self.compare_strategies_ttest(
                strategy_returns[name_a],
                strategy_returns[name_b],
                name_a,
                name_b
            )
            
            results.append({
                '策略 A': name_a,
                '策略 B': name_b,
                't 統計量': result.t_statistic,
                'p 值': result.p_value,
                '調整後 α': round(adjusted_alpha, 4),
                '顯著': '是' if result.p_value < adjusted_alpha else '否',
                '結論': result.conclusion,
            })
        
        return pd.DataFrame(results)
    
    def summary_statistics(self, returns: pd.Series) -> Dict[str, float]:
        """
        計算報酬序列的摘要統計
        
        Args:
            returns: 報酬序列
            
        Returns:
            統計摘要字典
        """
        data = returns.dropna()
        
        if len(data) < 2:
            return {'error': '數據不足'}
        
        return {
            '樣本數': len(data),
            '平均值': round(data.mean(), 4),
            '中位數': round(data.median(), 4),
            '標準差': round(data.std(), 4),
            '最小值': round(data.min(), 4),
            '最大值': round(data.max(), 4),
            '偏度': round(stats.skew(data), 4),
            '峰度': round(stats.kurtosis(data), 4),
            '正報酬比例': round((data > 0).mean() * 100, 2),
        }
