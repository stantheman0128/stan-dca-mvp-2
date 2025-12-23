# strategies/dca_dip_buying.py
"""V1: 跌深加碼策略"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_strategy import BaseStrategy, InvestmentDecision


class DCADipBuyingStrategy(BaseStrategy):
    """
    V1: 跌深加碼策略
    
    當價格相對近期高點大幅下跌時，增加投入金額：
    - 追蹤過去 N 天的最高價
    - 根據跌幅程度決定加碼倍數
    - 跌越多、投越多
    """
    
    def __init__(
        self,
        base_amount: float = 10000.0,
        lookback_period: int = 252,
        dip_threshold_1: float = 0.10,
        multiplier_1: float = 1.5,
        dip_threshold_2: float = 0.20,
        multiplier_2: float = 2.0
    ):
        """
        初始化跌深加碼策略
        
        Args:
            base_amount: 每期基礎投入金額
            lookback_period: 高點回顧期間（交易日，預設 252 約 1 年）
            dip_threshold_1: 第一級跌幅閾值（預設 10%）
            multiplier_1: 第一級加碼倍數（預設 1.5x）
            dip_threshold_2: 第二級跌幅閾值（預設 20%）
            multiplier_2: 第二級加碼倍數（預設 2.0x）
        """
        super().__init__(base_amount)
        self.lookback_period = lookback_period
        self.dip_threshold_1 = dip_threshold_1
        self.multiplier_1 = multiplier_1
        self.dip_threshold_2 = dip_threshold_2
        self.multiplier_2 = multiplier_2
    
    @property
    def name(self) -> str:
        return "V1: 跌深加碼"
    
    @property
    def description(self) -> str:
        return f"價格下跌 {self.dip_threshold_1*100:.0f}% 加碼 {self.multiplier_1}x，下跌 {self.dip_threshold_2*100:.0f}% 加碼 {self.multiplier_2}x"
    
    def calculate_investment(
        self,
        current_date: pd.Timestamp,
        current_price: float,
        historical_data: pd.DataFrame,
        portfolio_state: Dict[str, Any]
    ) -> InvestmentDecision:
        """
        跌深加碼策略：根據相對高點跌幅決定投入倍數
        """
        # 取得回顧期間的數據
        lookback_data = historical_data.tail(self.lookback_period)
        
        if len(lookback_data) < 5:
            # 數據不足，使用基礎金額
            return InvestmentDecision(
                base_amount=self.base_amount,
                multiplier=1.0,
                final_amount=self.base_amount,
                reason="數據不足，使用基礎投入"
            )
        
        # 計算近期高點
        recent_high = lookback_data['Close'].max()
        
        # 計算當前跌幅
        dip_pct = (recent_high - current_price) / recent_high
        
        # 決定投入倍數
        if dip_pct >= self.dip_threshold_2:
            multiplier = self.multiplier_2
            reason = f"跌幅 {dip_pct*100:.1f}% ≥ {self.dip_threshold_2*100:.0f}%，加碼 {self.multiplier_2}x"
        elif dip_pct >= self.dip_threshold_1:
            multiplier = self.multiplier_1
            reason = f"跌幅 {dip_pct*100:.1f}% ≥ {self.dip_threshold_1*100:.0f}%，加碼 {self.multiplier_1}x"
        else:
            multiplier = 1.0
            reason = f"跌幅 {dip_pct*100:.1f}% < {self.dip_threshold_1*100:.0f}%，正常投入"
        
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
            'lookback_period': self.lookback_period,
            'dip_threshold_1': self.dip_threshold_1,
            'multiplier_1': self.multiplier_1,
            'dip_threshold_2': self.dip_threshold_2,
            'multiplier_2': self.multiplier_2,
        }
