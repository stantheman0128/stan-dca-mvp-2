# strategies/base_strategy.py
"""策略基礎類別"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import pandas as pd


@dataclass
class StrategyConfig:
    """策略配置"""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class InvestmentDecision:
    """投資決策"""
    base_amount: float  # 基礎投入金額
    multiplier: float  # 投入倍數
    final_amount: float  # 最終投入金額
    reason: str  # 決策原因


class BaseStrategy(ABC):
    """
    策略基礎抽象類別
    
    所有 DCA 策略都應繼承此類別並實現 calculate_investment 方法
    """
    
    def __init__(self, base_amount: float = 10000.0):
        """
        初始化策略
        
        Args:
            base_amount: 每期基礎投入金額
        """
        self.base_amount = base_amount
        self._config: Optional[StrategyConfig] = None
        
    @property
    @abstractmethod
    def name(self) -> str:
        """策略名稱"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """策略描述"""
        pass
    
    @abstractmethod
    def calculate_investment(
        self,
        current_date: pd.Timestamp,
        current_price: float,
        historical_data: pd.DataFrame,
        portfolio_state: Dict[str, Any]
    ) -> InvestmentDecision:
        """
        計算當期投資金額
        
        Args:
            current_date: 當前日期
            current_price: 當前價格
            historical_data: 歷史數據（截至當前日期）
            portfolio_state: 投資組合狀態 {
                'total_shares': 累積持股,
                'total_cost': 累積成本,
                'current_value': 當前市值
            }
            
        Returns:
            InvestmentDecision 物件
        """
        pass
    
    def get_config(self) -> StrategyConfig:
        """取得策略配置"""
        if self._config is None:
            self._config = StrategyConfig(
                name=self.name,
                description=self.description,
                parameters=self.get_parameters()
            )
        return self._config
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """取得策略參數"""
        pass
    
    def set_parameters(self, **kwargs):
        """設定策略參數"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
