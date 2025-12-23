# utils/robustness_tester.py
"""穩健性測試模組"""
import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import random

from strategies.base_strategy import BaseStrategy
from core.backtest_engine import BacktestEngine, BacktestResult
from core.metrics import MetricsCalculator


@dataclass
class RobustnessTestResult:
    """穩健性測試結果"""
    test_name: str
    strategy_name: str
    results: List[Dict[str, Any]]
    summary: Dict[str, float]


class RobustnessTester:
    """
    穩健性測試器
    
    提供跨時間、跨市場、Monte Carlo 模擬等穩健性測試
    """
    
    # 預設測試起始點
    DEFAULT_START_POINTS = [
        date(2005, 12, 23),  # 完整 20 年
        date(2008, 1, 1),    # 金融海嘯前
        date(2009, 3, 1),    # 海嘯後谷底
        date(2015, 1, 1),    # 中期
        date(2020, 1, 1),    # 疫情前
        date(2020, 4, 1),    # 疫情後
    ]
    
    def __init__(
        self,
        engine: Optional[BacktestEngine] = None,
        max_workers: int = 4
    ):
        """
        初始化穩健性測試器
        
        Args:
            engine: 回測引擎
            max_workers: 最大並行工作數
        """
        self.engine = engine or BacktestEngine()
        self.max_workers = max_workers
    
    def test_fixed_start_points(
        self,
        strategy: BaseStrategy,
        market_data: pd.DataFrame,
        symbol: str,
        start_points: Optional[List[date]] = None,
        end_date: Optional[date] = None,
        frequency: str = 'M',
        progress_callback: Optional[callable] = None
    ) -> RobustnessTestResult:
        """
        固定起始點測試
        
        Args:
            strategy: 策略實例
            market_data: 市場數據
            symbol: 股票代碼
            start_points: 起始日期列表（預設使用 DEFAULT_START_POINTS）
            end_date: 結束日期（預設今天）
            frequency: 投資頻率
            progress_callback: 進度回調函數 (current, total)
            
        Returns:
            RobustnessTestResult 物件
        """
        if start_points is None:
            start_points = self.DEFAULT_START_POINTS
        if end_date is None:
            end_date = date.today()
        
        results = []
        total = len(start_points)
        
        for i, start in enumerate(start_points):
            try:
                result = self.engine.run(
                    strategy=strategy,
                    market_data=market_data,
                    symbol=symbol,
                    start_date=start,
                    end_date=end_date,
                    frequency=frequency
                )
                
                results.append({
                    'start_date': start,
                    'end_date': end_date,
                    'total_return': result.metrics.total_return,
                    'cagr': result.metrics.cagr,
                    'max_drawdown': result.metrics.max_drawdown,
                    'sharpe_ratio': result.metrics.sharpe_ratio,
                    'months': result.metrics.investment_months,
                    'success': True,
                })
            except Exception as e:
                results.append({
                    'start_date': start,
                    'end_date': end_date,
                    'error': str(e),
                    'success': False,
                })
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        # 計算摘要統計
        successful = [r for r in results if r.get('success', False)]
        summary = self._calculate_summary(successful)
        
        return RobustnessTestResult(
            test_name="固定起始點測試",
            strategy_name=strategy.name,
            results=results,
            summary=summary
        )
    
    def monte_carlo_simulation(
        self,
        strategy: BaseStrategy,
        market_data: pd.DataFrame,
        symbol: str,
        num_simulations: int = 300,
        min_duration_years: float = 3,
        max_duration_years: float = 15,
        frequency: str = 'M',
        progress_callback: Optional[callable] = None
    ) -> RobustnessTestResult:
        """
        Monte Carlo 隨機起始點模擬
        
        Args:
            strategy: 策略實例
            market_data: 市場數據
            symbol: 股票代碼
            num_simulations: 模擬次數
            min_duration_years: 最短投資期間（年）
            max_duration_years: 最長投資期間（年）
            frequency: 投資頻率
            progress_callback: 進度回調函數 (current, total)
            
        Returns:
            RobustnessTestResult 物件
        """
        # 確定可用日期範圍
        data_start = market_data.index.min().date()
        data_end = market_data.index.max().date()
        
        # 計算最早可用的起始日期（確保有足夠數據）
        min_duration_days = int(min_duration_years * 365)
        latest_start = data_end - timedelta(days=min_duration_days)
        
        if latest_start <= data_start:
            raise ValueError("數據範圍不足以進行 Monte Carlo 模擬")
        
        results = []
        
        def run_single_simulation(seed: int) -> Dict[str, Any]:
            """執行單次模擬"""
            random.seed(seed)
            
            # 隨機選擇起始日期
            days_range = (latest_start - data_start).days
            random_days = random.randint(0, days_range)
            start = data_start + timedelta(days=random_days)
            
            # 隨機選擇投資期間
            duration_years = random.uniform(min_duration_years, max_duration_years)
            duration_days = int(duration_years * 365)
            end = min(start + timedelta(days=duration_days), data_end)
            
            try:
                result = self.engine.run(
                    strategy=strategy,
                    market_data=market_data,
                    symbol=symbol,
                    start_date=start,
                    end_date=end,
                    frequency=frequency
                )
                
                return {
                    'start_date': start,
                    'end_date': end,
                    'duration_years': duration_years,
                    'total_return': result.metrics.total_return,
                    'cagr': result.metrics.cagr,
                    'max_drawdown': result.metrics.max_drawdown,
                    'sharpe_ratio': result.metrics.sharpe_ratio,
                    'success': True,
                }
            except Exception as e:
                return {
                    'start_date': start,
                    'end_date': end,
                    'error': str(e),
                    'success': False,
                }
        
        # 並行執行模擬
        completed = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(run_single_simulation, i): i 
                      for i in range(num_simulations)}
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, num_simulations)
        
        # 計算摘要統計
        successful = [r for r in results if r.get('success', False)]
        summary = self._calculate_summary(successful)
        
        # 額外計算勝率
        if successful:
            positive_returns = sum(1 for r in successful if r['total_return'] > 0)
            summary['win_rate'] = round(positive_returns / len(successful) * 100, 2)
        
        return RobustnessTestResult(
            test_name="Monte Carlo 模擬",
            strategy_name=strategy.name,
            results=results,
            summary=summary
        )
    
    def rolling_window_analysis(
        self,
        strategy: BaseStrategy,
        market_data: pd.DataFrame,
        symbol: str,
        window_years: int = 5,
        step_months: int = 1,
        frequency: str = 'M',
        progress_callback: Optional[callable] = None
    ) -> RobustnessTestResult:
        """
        滾動窗口分析
        
        Args:
            strategy: 策略實例
            market_data: 市場數據
            symbol: 股票代碼
            window_years: 窗口大小（年）
            step_months: 滾動步長（月）
            frequency: 投資頻率
            progress_callback: 進度回調函數 (current, total)
            
        Returns:
            RobustnessTestResult 物件
        """
        data_start = market_data.index.min().date()
        data_end = market_data.index.max().date()
        
        window_days = window_years * 365
        step_days = step_months * 30
        
        # 生成所有窗口
        windows = []
        current_start = data_start
        while True:
            current_end = current_start + timedelta(days=window_days)
            if current_end > data_end:
                break
            windows.append((current_start, current_end))
            current_start = current_start + timedelta(days=step_days)
        
        if not windows:
            raise ValueError("數據範圍不足以進行滾動窗口分析")
        
        results = []
        total = len(windows)
        
        for i, (start, end) in enumerate(windows):
            try:
                result = self.engine.run(
                    strategy=strategy,
                    market_data=market_data,
                    symbol=symbol,
                    start_date=start,
                    end_date=end,
                    frequency=frequency
                )
                
                results.append({
                    'start_date': start,
                    'end_date': end,
                    'total_return': result.metrics.total_return,
                    'cagr': result.metrics.cagr,
                    'max_drawdown': result.metrics.max_drawdown,
                    'sharpe_ratio': result.metrics.sharpe_ratio,
                    'success': True,
                })
            except Exception as e:
                results.append({
                    'start_date': start,
                    'end_date': end,
                    'error': str(e),
                    'success': False,
                })
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        # 計算摘要統計
        successful = [r for r in results if r.get('success', False)]
        summary = self._calculate_summary(successful)
        
        return RobustnessTestResult(
            test_name=f"滾動窗口分析 ({window_years}年窗口)",
            strategy_name=strategy.name,
            results=results,
            summary=summary
        )
    
    def _calculate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """計算摘要統計"""
        if not results:
            return {'error': '無有效結果'}
        
        returns = [r['total_return'] for r in results]
        cagrs = [r['cagr'] for r in results]
        drawdowns = [r['max_drawdown'] for r in results]
        sharpes = [r['sharpe_ratio'] for r in results]
        
        return {
            'count': len(results),
            'return_mean': round(np.mean(returns), 2),
            'return_median': round(np.median(returns), 2),
            'return_std': round(np.std(returns), 2),
            'return_min': round(np.min(returns), 2),
            'return_max': round(np.max(returns), 2),
            'return_p5': round(np.percentile(returns, 5), 2),
            'return_p95': round(np.percentile(returns, 95), 2),
            'cagr_mean': round(np.mean(cagrs), 2),
            'drawdown_mean': round(np.mean(drawdowns), 2),
            'drawdown_worst': round(np.min(drawdowns), 2),
            'sharpe_mean': round(np.mean(sharpes), 2),
        }
