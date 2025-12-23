# core/backtest_engine.py
"""回測引擎模組"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import date

from strategies.base_strategy import BaseStrategy, InvestmentDecision
from .metrics import MetricsCalculator, PerformanceMetrics


@dataclass
class Transaction:
    """交易記錄"""
    date: pd.Timestamp
    price: float
    investment_amount: float
    shares_bought: float
    cumulative_shares: float
    cumulative_cost: float
    market_value: float
    return_pct: float
    multiplier: float
    reason: str


@dataclass
class BacktestResult:
    """回測結果"""
    # 策略資訊
    strategy_name: str
    strategy_config: Dict[str, Any]
    
    # 市場資訊
    symbol: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    frequency: str
    
    # 交易記錄
    transactions: List[Transaction]
    
    # 權益曲線
    equity_curve: pd.DataFrame
    
    # 績效指標
    metrics: PerformanceMetrics
    
    # 最終狀態
    total_shares: float
    total_invested: float
    final_value: float
    
    def to_transactions_df(self) -> pd.DataFrame:
        """轉換交易記錄為 DataFrame"""
        if not self.transactions:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                '日期': t.date,
                '價格': t.price,
                '投入金額': t.investment_amount,
                '買入股數': t.shares_bought,
                '累積股數': t.cumulative_shares,
                '累積成本': t.cumulative_cost,
                '市值': t.market_value,
                '報酬率(%)': t.return_pct,
                '投入倍數': t.multiplier,
                '決策原因': t.reason,
            }
            for t in self.transactions
        ])


class BacktestEngine:
    """
    回測引擎
    
    負責執行策略回測、記錄交易、計算績效
    """
    
    FREQUENCY_MAP = {
        'W': 'W-MON',   # 每週（週一）
        'M': 'MS',       # 每月（月初）
        'Q': 'QS',       # 每季（季初）
    }
    
    def __init__(self, metrics_calculator: Optional[MetricsCalculator] = None):
        """
        初始化回測引擎
        
        Args:
            metrics_calculator: 指標計算器（可選）
        """
        self.metrics_calculator = metrics_calculator or MetricsCalculator()
    
    def run(
        self,
        strategy: BaseStrategy,
        market_data: pd.DataFrame,
        symbol: str = 'Unknown',
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        frequency: str = 'M'
    ) -> BacktestResult:
        """
        執行回測
        
        Args:
            strategy: 策略實例
            market_data: 市場數據 DataFrame（需包含 Close 欄位）
            symbol: 股票代碼
            start_date: 回測開始日期（預設數據起始）
            end_date: 回測結束日期（預設數據結束）
            frequency: 投資頻率（'W'=週, 'M'=月, 'Q'=季）
            
        Returns:
            BacktestResult 物件
        """
        # 準備數據
        data = self._prepare_data(market_data, start_date, end_date)
        
        # 根據頻率重採樣（取每期第一個交易日）
        resample_freq = self.FREQUENCY_MAP.get(frequency, 'MS')
        investment_dates = data['Close'].resample(resample_freq).first().dropna()
        
        if len(investment_dates) < 2:
            raise ValueError("數據不足，無法執行回測")
        
        # 初始化投資組合狀態
        total_shares = 0.0
        total_cost = 0.0
        transactions: List[Transaction] = []
        equity_records: List[Dict] = []
        
        # 逐期模擬
        for invest_date, price in investment_dates.items():
            price_val = float(price)
            
            # 取得截至當前的歷史數據
            historical = data.loc[:invest_date]
            
            # 計算當前市值
            current_value = total_shares * price_val
            
            # 投資組合狀態
            portfolio_state = {
                'total_shares': total_shares,
                'total_cost': total_cost,
                'current_value': current_value,
            }
            
            # 策略決策
            decision = strategy.calculate_investment(
                current_date=invest_date,
                current_price=price_val,
                historical_data=historical,
                portfolio_state=portfolio_state
            )
            
            # 執行買入
            shares_bought = decision.final_amount / price_val
            total_shares += shares_bought
            total_cost += decision.final_amount
            
            # 更新市值和報酬率
            new_value = total_shares * price_val
            return_pct = (new_value - total_cost) / total_cost * 100 if total_cost > 0 else 0
            
            # 記錄交易
            transactions.append(Transaction(
                date=invest_date,
                price=price_val,
                investment_amount=decision.final_amount,
                shares_bought=shares_bought,
                cumulative_shares=total_shares,
                cumulative_cost=total_cost,
                market_value=new_value,
                return_pct=return_pct,
                multiplier=decision.multiplier,
                reason=decision.reason
            ))
            
            # 記錄權益
            equity_records.append({
                'date': invest_date,
                'value': new_value,
                'cost': total_cost,
                'return_pct': return_pct
            })
        
        # 建立權益曲線
        equity_curve = pd.DataFrame(equity_records)
        
        # 計算績效指標
        periods_per_year = {'W': 52, 'M': 12, 'Q': 4}.get(frequency, 12)
        metrics = self.metrics_calculator.calculate_all(
            equity_curve=equity_curve,
            total_invested=total_cost,
            total_shares=total_shares,
            periods_per_year=periods_per_year
        )
        
        # 建立結果
        return BacktestResult(
            strategy_name=strategy.name,
            strategy_config=strategy.get_parameters(),
            symbol=symbol,
            start_date=equity_curve['date'].iloc[0],
            end_date=equity_curve['date'].iloc[-1],
            frequency=frequency,
            transactions=transactions,
            equity_curve=equity_curve,
            metrics=metrics,
            total_shares=total_shares,
            total_invested=total_cost,
            final_value=equity_curve['value'].iloc[-1]
        )
    
    def _prepare_data(
        self,
        data: pd.DataFrame,
        start_date: Optional[date],
        end_date: Optional[date]
    ) -> pd.DataFrame:
        """準備數據：篩選範圍、處理缺失值"""
        df = data.copy()
        
        # 確保索引是 DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # 篩選日期範圍
        if start_date:
            df = df[df.index.date >= start_date]
        if end_date:
            df = df[df.index.date <= end_date]
        
        # 前向填充缺失值
        df = df.ffill()
        
        return df
    
    def run_multiple(
        self,
        strategies: List[BaseStrategy],
        market_data: pd.DataFrame,
        symbol: str = 'Unknown',
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        frequency: str = 'M'
    ) -> Dict[str, BacktestResult]:
        """
        執行多策略回測
        
        Args:
            strategies: 策略列表
            其他參數同 run()
            
        Returns:
            {策略名稱: BacktestResult} 字典
        """
        results = {}
        for strategy in strategies:
            result = self.run(
                strategy=strategy,
                market_data=market_data,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                frequency=frequency
            )
            results[strategy.name] = result
        
        return results
