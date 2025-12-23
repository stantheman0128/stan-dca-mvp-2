# strategies/dca_pure.py
"""V0: 純定期定額策略"""
import pandas as pd
from typing import Dict, Any
from .base_strategy import BaseStrategy, InvestmentDecision


class DCAPureStrategy(BaseStrategy):
    """
    V0: 純定期定額策略（基準線）
    
    最基本的定期定額策略：
    - 每期固定投入相同金額
    - 不考慮市場狀況
    - 作為所有優化策略的對比基準
    """
    
    @property
    def name(self) -> str:
        return "V0: 純定期定額"
    
    @property
    def description(self) -> str:
        return "每期固定投入相同金額，不考慮市場狀況，作為基準策略"
    
    def calculate_investment(
        self,
        current_date: pd.Timestamp,
        current_price: float,
        historical_data: pd.DataFrame,
        portfolio_state: Dict[str, Any]
    ) -> InvestmentDecision:
        """
        純 DCA 策略：永遠投入固定金額
        """
        return InvestmentDecision(
            base_amount=self.base_amount,
            multiplier=1.0,
            final_amount=self.base_amount,
            reason="定期定額固定投入"
        )
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'base_amount': self.base_amount,
        }
