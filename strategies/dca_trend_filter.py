# strategies/dca_trend_filter.py
"""V2: 趨勢過濾策略"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_strategy import BaseStrategy, InvestmentDecision


class DCATrendFilterStrategy(BaseStrategy):
    """
    V2: 趨勢過濾策略
    
    根據價格與長期移動平均線的關係調整投入：
    - 計算長期移動平均線（如 200 日均線）
    - 價格低於均線時加碼（逢低買入）
    - 價格高於均線時正常投入
    """
    
    def __init__(
        self,
        base_amount: float = 10000.0,
        ma_period: int = 200,
        ma_type: str = 'SMA',
        above_multiplier: float = 1.0,
        below_multiplier: float = 1.5
    ):
        """
        初始化趨勢過濾策略
        
        Args:
            base_amount: 每期基礎投入金額
            ma_period: 移動平均週期（預設 200 日）
            ma_type: 移動平均類型（'SMA' 或 'EMA'）
            above_multiplier: 價格在 MA 上方時的投入倍數（預設 1.0）
            below_multiplier: 價格在 MA 下方時的投入倍數（預設 1.5）
        """
        super().__init__(base_amount)
        self.ma_period = ma_period
        self.ma_type = ma_type
        self.above_multiplier = above_multiplier
        self.below_multiplier = below_multiplier
    
    @property
    def name(self) -> str:
        return "V2: 趨勢過濾"
    
    @property
    def description(self) -> str:
        return f"價格低於 {self.ma_type}{self.ma_period} 時加碼 {self.below_multiplier}x"
    
    def _calculate_ma(self, data: pd.Series) -> float:
        """計算移動平均"""
        if len(data) < self.ma_period:
            return data.mean()
        
        if self.ma_type == 'EMA':
            return data.ewm(span=self.ma_period, adjust=False).mean().iloc[-1]
        else:  # SMA
            return data.tail(self.ma_period).mean()
    
    def calculate_investment(
        self,
        current_date: pd.Timestamp,
        current_price: float,
        historical_data: pd.DataFrame,
        portfolio_state: Dict[str, Any]
    ) -> InvestmentDecision:
        """
        趨勢過濾策略：價格低於均線時加碼
        """
        if len(historical_data) < 20:
            return InvestmentDecision(
                base_amount=self.base_amount,
                multiplier=1.0,
                final_amount=self.base_amount,
                reason="數據不足，使用基礎投入"
            )
        
        # 計算移動平均
        ma_value = self._calculate_ma(historical_data['Close'])
        
        # 決定投入倍數
        if current_price < ma_value:
            multiplier = self.below_multiplier
            diff_pct = (ma_value - current_price) / ma_value * 100
            reason = f"價格 {current_price:.2f} < {self.ma_type}{self.ma_period} ({ma_value:.2f})，低於 {diff_pct:.1f}%，加碼 {self.below_multiplier}x"
        else:
            multiplier = self.above_multiplier
            diff_pct = (current_price - ma_value) / ma_value * 100
            reason = f"價格 {current_price:.2f} ≥ {self.ma_type}{self.ma_period} ({ma_value:.2f})，高於 {diff_pct:.1f}%，正常投入"
        
        final_amount = self.base_amount * multiplier
        
        return InvestmentDecision(
            base_amount=self.base_amount,
            multiplier=multiplier,
            final_amount=final_amount,
            reason=reason
        )
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'base_amount': self.base_amount,
            'ma_period': self.ma_period,
            'ma_type': self.ma_type,
            'above_multiplier': self.above_multiplier,
            'below_multiplier': self.below_multiplier,
        }
