# utils/grid_search.py
"""
全自動參數網格搜索工具
Auto Grid Search for DCA Strategy Optimization
"""
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from core.data_loader import DataLoader
from core.backtest_engine import BacktestEngine
from core.metrics import MetricsCalculator
from strategies import (
    DCAPureStrategy,
    DCADipBuyingStrategy,
    DCATrendFilterStrategy,
    DCAVolatilityStrategy,
)


@dataclass
class GridSearchConfig:
    """網格搜索配置"""
    # 要測試的策略
    strategies: List[str] = None
    # 要測試的市場
    markets: List[str] = None
    # 時間區間
    start_dates: List[date] = None
    end_dates: List[date] = None
    # 基本投入金額
    base_amounts: List[float] = None
    # 投入頻率
    frequencies: List[str] = None
    
    # V1 參數範圍
    v1_dip_thresholds: List[Tuple[float, float]] = None  # (threshold_1, threshold_2)
    v1_multipliers: List[Tuple[float, float]] = None  # (multiplier_1, multiplier_2)
    v1_lookbacks: List[int] = None
    
    # V2 參數範圍
    v2_ma_periods: List[int] = None
    v2_ma_types: List[str] = None
    v2_below_multipliers: List[float] = None
    
    # V3 參數範圍
    v3_vol_windows: List[int] = None
    v3_vol_thresholds: List[Tuple[float, float]] = None  # (low, high)
    v3_vol_multipliers: List[Tuple[float, float]] = None  # (low_mult, high_mult)
    
    def __post_init__(self):
        """設定預設值"""
        if self.strategies is None:
            self.strategies = ['V0', 'V1', 'V2', 'V3']
        if self.markets is None:
            self.markets = ['SPY', 'QQQ', '0050.TW']
        if self.base_amounts is None:
            self.base_amounts = [10000]
        if self.frequencies is None:
            self.frequencies = ['monthly']
        if self.v1_dip_thresholds is None:
            self.v1_dip_thresholds = [(0.10, 0.20)]
        if self.v1_multipliers is None:
            self.v1_multipliers = [(1.5, 2.0)]
        if self.v1_lookbacks is None:
            self.v1_lookbacks = [252]
        if self.v2_ma_periods is None:
            self.v2_ma_periods = [200]
        if self.v2_ma_types is None:
            self.v2_ma_types = ['SMA']
        if self.v2_below_multipliers is None:
            self.v2_below_multipliers = [1.5]
        if self.v3_vol_windows is None:
            self.v3_vol_windows = [20]
        if self.v3_vol_thresholds is None:
            self.v3_vol_thresholds = [(0.8, 1.5)]
        if self.v3_vol_multipliers is None:
            self.v3_vol_multipliers = [(0.8, 1.5)]


class GridSearchOptimizer:
    """網格搜索優化器"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.backtest_engine = BacktestEngine()
        self.metrics_calculator = MetricsCalculator()
        # 只保留摘要結果，不保留完整 DataFrame
        self.results: List[Dict] = []
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._max_cache_size = 5  # 最多緩存 5 個市場的數據
    
    def _get_market_data(
        self, 
        symbol: str, 
        start_date: date, 
        end_date: date
    ) -> Optional[pd.DataFrame]:
        """取得市場資料（帶快取，限制緩存大小）"""
        cache_key = f"{symbol}_{start_date}_{end_date}"
        
        # 如果緩存太大，清除最舊的
        if len(self._data_cache) >= self._max_cache_size and cache_key not in self._data_cache:
            oldest_key = next(iter(self._data_cache))
            del self._data_cache[oldest_key]
        
        if cache_key not in self._data_cache:
            data = self.data_loader.download(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            if data is not None and len(data) > 0:
                self._data_cache[cache_key] = data
            else:
                return None
        return self._data_cache[cache_key]
    
    def clear_cache(self):
        """清除內部緩存"""
        self._data_cache.clear()
        self.results.clear()
    
    def _create_strategy(
        self,
        strategy_name: str,
        base_amount: float,
        params: Dict[str, Any]
    ):
        """創建策略實例"""
        if strategy_name == 'V0':
            return DCAPureStrategy(base_amount=base_amount)
        elif strategy_name == 'V1':
            return DCADipBuyingStrategy(
                base_amount=base_amount,
                lookback_period=params.get('lookback', 252),
                dip_threshold_1=params.get('dip_threshold_1', 0.10),
                multiplier_1=params.get('multiplier_1', 1.5),
                dip_threshold_2=params.get('dip_threshold_2', 0.20),
                multiplier_2=params.get('multiplier_2', 2.0),
            )
        elif strategy_name == 'V2':
            return DCATrendFilterStrategy(
                base_amount=base_amount,
                ma_period=params.get('ma_period', 200),
                ma_type=params.get('ma_type', 'SMA'),
                below_multiplier=params.get('below_multiplier', 1.5),
            )
        elif strategy_name == 'V3':
            return DCAVolatilityStrategy(
                base_amount=base_amount,
                volatility_window=params.get('vol_window', 20),
                high_vol_threshold=params.get('high_vol_threshold', 1.5),
                low_vol_threshold=params.get('low_vol_threshold', 0.8),
                high_vol_multiplier=params.get('high_vol_multiplier', 1.5),
                low_vol_multiplier=params.get('low_vol_multiplier', 0.8),
            )
        return DCAPureStrategy(base_amount=base_amount)
    
    def _get_strategy_params_combinations(
        self,
        strategy_name: str,
        config: GridSearchConfig
    ) -> List[Dict]:
        """取得策略的所有參數組合"""
        if strategy_name == 'V0':
            return [{}]  # V0 沒有額外參數
        
        elif strategy_name == 'V1':
            combinations = []
            for lookback in config.v1_lookbacks:
                for dip_thresh in config.v1_dip_thresholds:
                    for mult in config.v1_multipliers:
                        combinations.append({
                            'lookback': lookback,
                            'dip_threshold_1': dip_thresh[0],
                            'dip_threshold_2': dip_thresh[1],
                            'multiplier_1': mult[0],
                            'multiplier_2': mult[1],
                        })
            return combinations
        
        elif strategy_name == 'V2':
            combinations = []
            for ma_period in config.v2_ma_periods:
                for ma_type in config.v2_ma_types:
                    for below_mult in config.v2_below_multipliers:
                        combinations.append({
                            'ma_period': ma_period,
                            'ma_type': ma_type,
                            'below_multiplier': below_mult,
                        })
            return combinations
        
        elif strategy_name == 'V3':
            combinations = []
            for vol_window in config.v3_vol_windows:
                for vol_thresh in config.v3_vol_thresholds:
                    for vol_mult in config.v3_vol_multipliers:
                        combinations.append({
                            'vol_window': vol_window,
                            'low_vol_threshold': vol_thresh[0],
                            'high_vol_threshold': vol_thresh[1],
                            'low_vol_multiplier': vol_mult[0],
                            'high_vol_multiplier': vol_mult[1],
                        })
            return combinations
        
        return [{}]
    
    def count_total_combinations(self, config: GridSearchConfig) -> int:
        """計算總測試組合數"""
        total = 0
        n_time_periods = len(config.start_dates) if config.start_dates else 1
        n_amounts = len(config.base_amounts)
        n_frequencies = len(config.frequencies)
        n_markets = len(config.markets)
        
        for strategy in config.strategies:
            n_params = len(self._get_strategy_params_combinations(strategy, config))
            total += n_markets * n_time_periods * n_amounts * n_frequencies * n_params
        
        return total
    
    def run_grid_search(
        self,
        config: GridSearchConfig,
        progress_callback=None
    ) -> pd.DataFrame:
        """執行網格搜索"""
        self.results = []
        
        # 計算總組合數
        total_combinations = self.count_total_combinations(config)
        current = 0
        
        # 設定預設時間區間
        if config.start_dates is None:
            config.start_dates = [date(2010, 1, 1)]
        if config.end_dates is None:
            config.end_dates = [date(2024, 12, 31)]
        
        # 遍歷所有組合
        for strategy_name in config.strategies:
            params_list = self._get_strategy_params_combinations(strategy_name, config)
            
            for market in config.markets:
                for i, start_dt in enumerate(config.start_dates):
                    end_dt = config.end_dates[i] if i < len(config.end_dates) else config.end_dates[-1]
                    
                    # 取得市場資料
                    market_data = self._get_market_data(market, start_dt, end_dt)
                    if market_data is None or len(market_data) < 100:
                        current += len(params_list) * len(config.base_amounts) * len(config.frequencies)
                        continue
                    
                    for base_amount in config.base_amounts:
                        for frequency in config.frequencies:
                            for params in params_list:
                                current += 1
                                
                                if progress_callback:
                                    progress_callback(current, total_combinations)
                                
                                try:
                                    # 創建策略
                                    strategy = self._create_strategy(
                                        strategy_name, base_amount, params
                                    )
                                    
                                    # 執行回測 (方法名稱是 run 不是 run_backtest)
                                    result = self.backtest_engine.run(
                                        strategy=strategy,
                                        market_data=market_data,
                                        symbol=market,
                                        frequency=frequency
                                    )
                                    
                                    if result is None:
                                        continue
                                    
                                    # BacktestResult 已經包含 metrics，直接使用
                                    metrics = result.metrics
                                    
                                    # 記錄結果
                                    result_record = {
                                        # 基本配置
                                        '策略': strategy_name,
                                        '市場': market,
                                        '開始日期': start_dt.strftime('%Y-%m-%d'),
                                        '結束日期': end_dt.strftime('%Y-%m-%d'),
                                        '投資年數': metrics.investment_years,
                                        '基本金額': base_amount,
                                        '頻率': frequency,
                                        
                                        # 核心績效
                                        '總投入': result.total_invested,
                                        '最終價值': result.final_value,
                                        '總報酬率(%)': round(metrics.total_return, 2),
                                        'CAGR(%)': round(metrics.cagr, 2),
                                        '夏普比率': round(metrics.sharpe_ratio, 3),
                                        '最大回撤(%)': round(metrics.max_drawdown, 2),
                                        '波動率(%)': round(metrics.volatility, 2),
                                        
                                        # 其他指標
                                        '勝率(%)': round(metrics.win_rate, 2),
                                        '總買進次數': metrics.total_periods,
                                        '平均成本': round(metrics.avg_cost, 2),
                                        
                                        # 策略參數（字串格式）
                                        '參數': str(params) if params else 'N/A',
                                    }
                                    
                                    self.results.append(result_record)
                                    
                                except Exception as e:
                                    # 記錄錯誤但繼續執行
                                    print(f"Error: {strategy_name} on {market}: {e}")
                                    continue
        
        # 轉換為 DataFrame
        df = pd.DataFrame(self.results)
        
        if len(df) > 0:
            # 排序：依 Sharpe Ratio 降序
            df = df.sort_values('夏普比率', ascending=False).reset_index(drop=True)
            df.index = df.index + 1  # 排名從 1 開始
            df.index.name = '排名'
        
        return df
    
    def get_best_by_metric(
        self,
        df: pd.DataFrame,
        metric: str = '夏普比率',
        top_n: int = 10
    ) -> pd.DataFrame:
        """取得特定指標的最佳組合"""
        ascending = metric in ['最大回撤(%)', '波動率(%)']  # 這些越小越好
        return df.nlargest(top_n, metric) if not ascending else df.nsmallest(top_n, metric)
    
    def get_summary_by_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """依策略彙總統計"""
        if len(df) == 0:
            return pd.DataFrame()
        
        summary = df.groupby('策略').agg({
            '總報酬率(%)': ['mean', 'std', 'min', 'max'],
            'CAGR(%)': ['mean', 'std', 'min', 'max'],
            '夏普比率': ['mean', 'std', 'min', 'max'],
            '最大回撤(%)': ['mean', 'std', 'min', 'max'],
        }).round(3)
        
        summary.columns = ['_'.join(col) for col in summary.columns]
        return summary
    
    def get_summary_by_market(self, df: pd.DataFrame) -> pd.DataFrame:
        """依市場彙總統計"""
        if len(df) == 0:
            return pd.DataFrame()
        
        summary = df.groupby('市場').agg({
            '總報酬率(%)': ['mean', 'std', 'min', 'max'],
            'CAGR(%)': ['mean', 'std', 'min', 'max'],
            '夏普比率': ['mean', 'std', 'min', 'max'],
            '最大回撤(%)': ['mean', 'std', 'min', 'max'],
        }).round(3)
        
        summary.columns = ['_'.join(col) for col in summary.columns]
        return summary
    
    def export_to_excel(self, df: pd.DataFrame, filename: str) -> bytes:
        """匯出結果到 Excel"""
        import io
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # 完整結果
            df.to_excel(writer, sheet_name='完整結果')
            
            # 最佳 Sharpe Ratio
            best_sharpe = self.get_best_by_metric(df, '夏普比率', 20)
            best_sharpe.to_excel(writer, sheet_name='最佳Sharpe')
            
            # 最佳總報酬
            best_return = self.get_best_by_metric(df, '總報酬率(%)', 20)
            best_return.to_excel(writer, sheet_name='最佳報酬')
            
            # 最低回撤
            best_dd = self.get_best_by_metric(df, '最大回撤(%)', 20)
            best_dd.to_excel(writer, sheet_name='最低回撤')
            
            # 策略彙總
            strategy_summary = self.get_summary_by_strategy(df)
            strategy_summary.to_excel(writer, sheet_name='策略彙總')
            
            # 市場彙總
            market_summary = self.get_summary_by_market(df)
            market_summary.to_excel(writer, sheet_name='市場彙總')
        
        return output.getvalue()


def create_default_config() -> GridSearchConfig:
    """創建預設配置（完整測試）"""
    return GridSearchConfig(
        strategies=['V0', 'V1', 'V2', 'V3'],
        markets=['SPY', 'QQQ', '0050.TW', 'VTI', 'IVV'],
        start_dates=[
            date(2010, 1, 1),
            date(2015, 1, 1),
            date(2018, 1, 1),
        ],
        end_dates=[
            date(2024, 12, 31),
            date(2024, 12, 31),
            date(2024, 12, 31),
        ],
        base_amounts=[10000],
        frequencies=['monthly'],
        v1_dip_thresholds=[(0.10, 0.20), (0.15, 0.25)],
        v1_multipliers=[(1.5, 2.0), (2.0, 3.0)],
        v1_lookbacks=[126, 252],
        v2_ma_periods=[100, 200],
        v2_ma_types=['SMA', 'EMA'],
        v2_below_multipliers=[1.3, 1.5, 2.0],
        v3_vol_windows=[20, 60],
        v3_vol_thresholds=[(0.7, 1.3), (0.8, 1.5)],
        v3_vol_multipliers=[(0.7, 1.5), (0.8, 2.0)],
    )


def create_quick_config() -> GridSearchConfig:
    """創建快速配置（少量測試）"""
    return GridSearchConfig(
        strategies=['V0', 'V1', 'V2', 'V3'],
        markets=['SPY', '0050.TW'],
        start_dates=[date(2015, 1, 1)],
        end_dates=[date(2024, 12, 31)],
        base_amounts=[10000],
        frequencies=['monthly'],
        v1_dip_thresholds=[(0.10, 0.20)],
        v1_multipliers=[(1.5, 2.0)],
        v1_lookbacks=[252],
        v2_ma_periods=[200],
        v2_ma_types=['SMA'],
        v2_below_multipliers=[1.5],
        v3_vol_windows=[20],
        v3_vol_thresholds=[(0.8, 1.5)],
        v3_vol_multipliers=[(0.8, 1.5)],
    )
