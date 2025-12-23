# core/metrics.py
"""績效指標計算模組"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """績效指標資料類別"""
    # 報酬指標
    total_return: float  # 總報酬率 (%)
    total_return_amount: float  # 總報酬金額
    cagr: float  # 年化報酬率 (%)
    
    # 風險指標
    max_drawdown: float  # 最大回撤 (%)
    max_drawdown_duration: int  # 最長回撤期間 (天)
    volatility: float  # 年化波動率 (%)
    downside_volatility: float  # 下行波動率 (%)
    var_95: float  # 95% VaR (%)
    var_99: float  # 99% VaR (%)
    
    # 風險調整報酬
    sharpe_ratio: float  # 夏普比率
    sortino_ratio: float  # 索提諾比率
    calmar_ratio: float  # 卡爾馬比率
    
    # 交易統計
    total_invested: float  # 總投入金額
    final_value: float  # 最終市值
    total_periods: int  # 總投入次數
    investment_months: int  # 投資月數
    avg_cost: float  # 平均持倉成本
    win_rate: float  # 勝率 (%)


class MetricsCalculator:
    """
    績效指標計算器
    
    計算所有報酬、風險和風險調整報酬指標
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        初始化計算器
        
        Args:
            risk_free_rate: 年化無風險利率（預設 2%）
        """
        self.risk_free_rate = risk_free_rate
        
    def calculate_all(
        self,
        equity_curve: pd.DataFrame,
        total_invested: float,
        total_shares: float,
        periods_per_year: int = 12
    ) -> PerformanceMetrics:
        """
        計算所有績效指標
        
        Args:
            equity_curve: 權益曲線 DataFrame，需包含 'value' 和 'date' 欄位
            total_invested: 總投入金額
            total_shares: 總持股數
            periods_per_year: 每年期數（月度=12，週度=52）
            
        Returns:
            PerformanceMetrics 物件
        """
        values = equity_curve['value'].values
        dates = pd.to_datetime(equity_curve['date'])
        
        final_value = values[-1]
        
        # 計算報酬序列
        returns = pd.Series(values).pct_change().dropna()
        
        # 報酬指標
        total_return = (final_value - total_invested) / total_invested * 100
        total_return_amount = final_value - total_invested
        
        years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
        if years > 0 and total_invested > 0:
            cagr = ((final_value / total_invested) ** (1 / years) - 1) * 100
        else:
            cagr = 0.0
            
        # 風險指標
        max_dd, max_dd_duration = self._calculate_drawdown(values, dates)
        volatility = self._calculate_volatility(returns, periods_per_year)
        downside_vol = self._calculate_downside_volatility(returns, periods_per_year)
        var_95 = self._calculate_var(returns, 0.05)
        var_99 = self._calculate_var(returns, 0.01)
        
        # 風險調整報酬
        sharpe = self._calculate_sharpe(returns, periods_per_year)
        sortino = self._calculate_sortino(returns, periods_per_year)
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0
        
        # 交易統計
        total_periods = len(equity_curve)
        investment_months = int(years * 12)
        avg_cost = total_invested / total_shares if total_shares > 0 else 0
        win_rate = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0
        
        return PerformanceMetrics(
            total_return=round(total_return, 2),
            total_return_amount=round(total_return_amount, 2),
            cagr=round(cagr, 2),
            max_drawdown=round(max_dd, 2),
            max_drawdown_duration=max_dd_duration,
            volatility=round(volatility, 2),
            downside_volatility=round(downside_vol, 2),
            var_95=round(var_95, 2),
            var_99=round(var_99, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            calmar_ratio=round(calmar, 2),
            total_invested=round(total_invested, 2),
            final_value=round(final_value, 2),
            total_periods=total_periods,
            investment_months=investment_months,
            avg_cost=round(avg_cost, 2),
            win_rate=round(win_rate, 2)
        )
    
    def _calculate_drawdown(
        self, 
        values: np.ndarray, 
        dates: pd.Series
    ) -> tuple[float, int]:
        """計算最大回撤和最長回撤期間"""
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak * 100
        max_dd = drawdown.min()
        
        # 計算最長回撤期間
        is_underwater = values < peak
        max_duration = 0
        current_duration = 0
        
        for i, underwater in enumerate(is_underwater):
            if underwater:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
                
        return max_dd, max_duration
    
    def _calculate_volatility(
        self, 
        returns: pd.Series, 
        periods_per_year: int
    ) -> float:
        """計算年化波動率"""
        if len(returns) < 2:
            return 0.0
        return returns.std() * np.sqrt(periods_per_year) * 100
    
    def _calculate_downside_volatility(
        self, 
        returns: pd.Series, 
        periods_per_year: int
    ) -> float:
        """計算下行波動率"""
        negative_returns = returns[returns < 0]
        if len(negative_returns) < 2:
            return 0.0
        return negative_returns.std() * np.sqrt(periods_per_year) * 100
    
    def _calculate_var(self, returns: pd.Series, alpha: float) -> float:
        """計算 Value at Risk"""
        if len(returns) < 10:
            return 0.0
        return np.percentile(returns, alpha * 100) * 100
    
    def _calculate_sharpe(
        self, 
        returns: pd.Series, 
        periods_per_year: int
    ) -> float:
        """計算夏普比率"""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
            
        excess_return = returns.mean() - self.risk_free_rate / periods_per_year
        return excess_return / returns.std() * np.sqrt(periods_per_year)
    
    def _calculate_sortino(
        self, 
        returns: pd.Series, 
        periods_per_year: int
    ) -> float:
        """計算索提諾比率"""
        negative_returns = returns[returns < 0]
        if len(negative_returns) < 2 or negative_returns.std() == 0:
            return 0.0
            
        excess_return = returns.mean() - self.risk_free_rate / periods_per_year
        downside_std = negative_returns.std()
        return excess_return / downside_std * np.sqrt(periods_per_year)
    
    def calculate_annual_returns(
        self, 
        equity_curve: pd.DataFrame
    ) -> Dict[int, float]:
        """計算每年報酬率"""
        df = equity_curve.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        
        annual_returns = {}
        years = df['year'].unique()
        
        for i, year in enumerate(sorted(years)):
            year_data = df[df['year'] == year]
            if len(year_data) >= 2:
                start_val = year_data['value'].iloc[0]
                end_val = year_data['value'].iloc[-1]
                annual_returns[year] = round((end_val - start_val) / start_val * 100, 2)
                
        return annual_returns
    
    def calculate_monthly_returns(
        self, 
        equity_curve: pd.DataFrame
    ) -> pd.DataFrame:
        """計算月度報酬率矩陣"""
        df = equity_curve.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        
        # 計算月度報酬
        df['return'] = df['value'].pct_change() * 100
        
        # 建立年-月矩陣
        monthly = df.groupby(['year', 'month'])['return'].sum().unstack()
        monthly.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(monthly.columns)]
        
        return monthly.round(2)
