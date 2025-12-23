# strategies/dca_volatility.py
"""V3: 波動率調整策略"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_strategy import BaseStrategy, InvestmentDecision


class DCAVolatilityStrategy(BaseStrategy):
    """
    V3: 波動率調整策略
    
    根據市場波動率調整投入：
    - 計算滾動波動率
    - 高波動（恐慌）時加碼
    - 低波動時減碼
    - 波動率 = 市場恐慌 = 買入機會
    """
    
    def __init__(
        self,
        base_amount: float = 10000.0,
        volatility_window: int = 20,
        lookback_period: int = 252,
        high_vol_threshold: float = 1.5,
        low_vol_threshold: float = 0.8,
        high_vol_multiplier: float = 1.5,
        low_vol_multiplier: float = 0.8
    ):
        """
        初始化波動率調整策略
        
        Args:
            base_amount: 每期基礎投入金額
            volatility_window: 波動率計算窗口（預設 20 日）
            lookback_period: 歷史平均波動率計算期間（預設 252 日）
            high_vol_threshold: 高波動閾值（預設 1.5 倍平均）
            low_vol_threshold: 低波動閾值（預設 0.8 倍平均）
            high_vol_multiplier: 高波動投入倍數（預設 1.5x）
            low_vol_multiplier: 低波動投入倍數（預設 0.8x）
        """
        super().__init__(base_amount)
        self.volatility_window = volatility_window
        self.lookback_period = lookback_period
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold
        self.high_vol_multiplier = high_vol_multiplier
        self.low_vol_multiplier = low_vol_multiplier
    
    @property
    def name(self) -> str:
        return "V3: 波動率調整"
    
    @property
    def description(self) -> str:
        return f"高波動(>{self.high_vol_threshold}x平均)加碼{self.high_vol_multiplier}x，低波動(<{self.low_vol_threshold}x平均)減碼{self.low_vol_multiplier}x"
    
    def _calculate_volatility(self, prices: pd.Series, window: int) -> float:
        """計算年化波動率"""
        if len(prices) < window + 1:
            return 0.0
        
        returns = prices.pct_change().dropna()
        if len(returns) < window:
            return returns.std() * np.sqrt(252)
            
        return returns.tail(window).std() * np.sqrt(252)
    
    def calculate_investment(
        self,
        current_date: pd.Timestamp,
        current_price: float,
        historical_data: pd.DataFrame,
        portfolio_state: Dict[str, Any]
    ) -> InvestmentDecision:
        """
        波動率調整策略：高波動加碼，低波動減碼
        """
        if len(historical_data) < self.volatility_window + 10:
            return InvestmentDecision(
                base_amount=self.base_amount,
                multiplier=1.0,
                final_amount=self.base_amount,
                reason="數據不足，使用基礎投入"
            )
        
        prices = historical_data['Close']
        
        # 計算當前波動率
        current_vol = self._calculate_volatility(prices, self.volatility_window)
        
        # 計算歷史平均波動率
        lookback_data = prices.tail(self.lookback_period)
        returns = lookback_data.pct_change().dropna()
        
        # 滾動計算波動率的平均值
        rolling_vol = returns.rolling(window=self.volatility_window).std() * np.sqrt(252)
        avg_vol = rolling_vol.mean()
        
        if avg_vol == 0 or np.isnan(avg_vol):
            return InvestmentDecision(
                base_amount=self.base_amount,
                multiplier=1.0,
                final_amount=self.base_amount,
                reason="無法計算波動率，使用基礎投入"
            )
        
        # 波動率比率
        vol_ratio = current_vol / avg_vol
        
        # 決定投入倍數
        if vol_ratio >= self.high_vol_threshold:
            multiplier = self.high_vol_multiplier
            reason = f"波動率 {current_vol*100:.1f}% = {vol_ratio:.2f}x 平均，高波動加碼 {self.high_vol_multiplier}x"
        elif vol_ratio <= self.low_vol_threshold:
            multiplier = self.low_vol_multiplier
            reason = f"波動率 {current_vol*100:.1f}% = {vol_ratio:.2f}x 平均，低波動減碼 {self.low_vol_multiplier}x"
        else:
            multiplier = 1.0
            reason = f"波動率 {current_vol*100:.1f}% = {vol_ratio:.2f}x 平均，正常投入"
        
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
            'volatility_window': self.volatility_window,
            'lookback_period': self.lookback_period,
            'high_vol_threshold': self.high_vol_threshold,
            'low_vol_threshold': self.low_vol_threshold,
            'high_vol_multiplier': self.high_vol_multiplier,
            'low_vol_multiplier': self.low_vol_multiplier,
        }
