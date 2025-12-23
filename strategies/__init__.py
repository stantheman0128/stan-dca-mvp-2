# strategies/__init__.py
from .base_strategy import BaseStrategy
from .dca_pure import DCAPureStrategy
from .dca_dip_buying import DCADipBuyingStrategy
from .dca_trend_filter import DCATrendFilterStrategy
from .dca_volatility import DCAVolatilityStrategy

__all__ = [
    'BaseStrategy',
    'DCAPureStrategy',
    'DCADipBuyingStrategy',
    'DCATrendFilterStrategy',
    'DCAVolatilityStrategy',
]

# 策略註冊表
STRATEGY_REGISTRY = {
    'V0: 純定期定額': DCAPureStrategy,
    'V1: 跌深加碼': DCADipBuyingStrategy,
    'V2: 趨勢過濾': DCATrendFilterStrategy,
    'V3: 波動率調整': DCAVolatilityStrategy,
}
