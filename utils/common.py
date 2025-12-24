# utils/common.py
"""
共用工具函數
提供跨頁面使用的輔助函數
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime
from typing import Dict, Any, List
from pathlib import Path
import pickle
import os
import gc

from core.data_loader import DataLoader
from core.backtest_engine import BacktestEngine
from core.metrics import MetricsCalculator
from core.statistics import StatisticsCalculator
from core.visualizer import Visualizer
from strategies import (
    STRATEGY_REGISTRY,
    DCAPureStrategy,
    DCADipBuyingStrategy,
    DCATrendFilterStrategy,
    DCAVolatilityStrategy,
)
from utils.report_generator import ReportGenerator
from utils.robustness_tester import RobustnessTester


# ===================== 緩存的資源獲取器 =====================
@st.cache_resource
def get_data_loader():
    """獲取數據載入器（單例）"""
    return DataLoader()

@st.cache_resource
def get_backtest_engine():
    """獲取回測引擎（單例）"""
    return BacktestEngine()

@st.cache_resource
def get_visualizer():
    """獲取視覺化器（單例）"""
    return Visualizer()

@st.cache_resource
def get_report_generator():
    """獲取報告生成器（單例）"""
    return ReportGenerator()

@st.cache_resource
def get_robustness_tester():
    """獲取穩健性測試器（單例）"""
    return RobustnessTester()


# ===================== 記憶體管理 =====================
def clear_memory_cache():
    """清除記憶體緩存"""
    # 清除 streamlit cache
    st.cache_data.clear()
    # 清除 session state 中的大型物件
    keys_to_remove = []
    for key in st.session_state:
        if any(x in key for x in ['results', 'data', 'grid_', 'backtest_']):
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del st.session_state[key]
    # 強制垃圾回收
    gc.collect()


def get_memory_usage():
    """取得當前記憶體使用量 (MB)"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0


# ===================== 策略工廠 =====================
def create_strategy_instance(
    strategy_name: str,
    base_amount: float,
    params: Dict[str, Any]
):
    """根據名稱創建策略實例"""
    if strategy_name == 'V0: 純定期定額':
        return DCAPureStrategy(base_amount=base_amount)
    elif strategy_name == 'V1: 跌深加碼':
        return DCADipBuyingStrategy(
            base_amount=base_amount,
            lookback_period=params.get('lookback_period', 252),
            dip_threshold_1=params.get('dip_threshold_1', 0.10),
            multiplier_1=params.get('multiplier_1', 1.5),
            dip_threshold_2=params.get('dip_threshold_2', 0.20),
            multiplier_2=params.get('multiplier_2', 2.0),
        )
    elif strategy_name == 'V2: 趨勢過濾':
        return DCATrendFilterStrategy(
            base_amount=base_amount,
            ma_period=params.get('ma_period', 200),
            ma_type=params.get('ma_type', 'SMA'),
            below_multiplier=params.get('below_multiplier', 1.5),
        )
    elif strategy_name == 'V3: 波動率調整':
        return DCAVolatilityStrategy(
            base_amount=base_amount,
            volatility_window=params.get('volatility_window', 20),
            high_vol_threshold=params.get('high_vol_threshold', 1.5),
            low_vol_threshold=params.get('low_vol_threshold', 0.8),
            high_vol_multiplier=params.get('high_vol_multiplier', 1.5),
            low_vol_multiplier=params.get('low_vol_multiplier', 0.8),
        )
    else:
        return DCAPureStrategy(base_amount=base_amount)


# ===================== UI 輔助函數 =====================
def render_metric_card(label: str, value: str, is_positive: bool = True):
    """渲染指標卡片"""
    color_class = "positive" if is_positive else "negative"
    st.markdown(f"""
    <div style="background: #f8f9fa; border-radius: 10px; padding: 15px; text-align: center; border-left: 4px solid {'#4CAF50' if is_positive else '#F44336'};">
        <div style="font-size: 0.85rem; color: #666; margin-bottom: 5px;">{label}</div>
        <div style="font-size: 1.5rem; font-weight: 700;" class="{color_class}">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def load_css():
    """載入外部 CSS 樣式"""
    css_path = Path(__file__).parent.parent / "assets" / "styles.css"
    if css_path.exists():
        with open(css_path, 'r', encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # 備用內嵌樣式
        st.markdown("""
        <style>
        .positive { color: #4CAF50; }
        .negative { color: #F44336; }
        </style>
        """, unsafe_allow_html=True)


def render_system_info():
    """渲染系統管理區塊（側邊欄）"""
    with st.expander("🔧 系統管理", expanded=False):
        mem_usage = get_memory_usage()
        if mem_usage > 0:
            st.metric("記憶體使用", f"{mem_usage:.1f} MB")
        else:
            st.info("安裝 psutil 可監控記憶體")
        
        if st.button("🗑️ 清除緩存", use_container_width=True):
            clear_memory_cache()
            st.success("✅ 緩存已清除")
            st.rerun()


# ===================== 計算輔助函數 =====================
def calculate_buy_and_hold(market_data: pd.DataFrame, total_investment: float) -> Dict:
    """計算 Buy & Hold 策略績效"""
    # 優先使用 Close，如果沒有則用 Adj Close
    price_col = 'Close' if 'Close' in market_data.columns else 'Adj Close'
    if price_col not in market_data.columns:
        raise KeyError(f"找不到價格欄位: {market_data.columns.tolist()}")
    
    prices = market_data[price_col]
    first_price = prices.iloc[0]
    last_price = prices.iloc[-1]
    shares = total_investment / first_price
    final_value = shares * last_price
    total_return = (final_value - total_investment) / total_investment * 100
    
    years = (market_data.index[-1] - market_data.index[0]).days / 365.25
    cagr = ((final_value / total_investment) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    # 計算回撤
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax * 100
    max_drawdown = drawdown.min()
    
    # 計算波動率
    returns = prices.pct_change().dropna()
    volatility = returns.std() * np.sqrt(252) * 100
    
    # 夏普比率
    sharpe = (cagr - 2) / volatility if volatility > 0 else 0
    
    return {
        'total_return': total_return,
        'cagr': cagr,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'volatility': volatility,
        'final_value': final_value,
        'total_invested': total_investment
    }


# ===================== 儲存/載入測試結果 =====================
def save_test_result(results: Dict, config: Dict, name: str) -> str:
    """保存測試結果"""
    save_dir = Path("results/saved_tests")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = save_dir / f"test_{timestamp}_{name}.pkl"
    
    data = {
        'results': results,
        'config': config,
        'name': name,
        'timestamp': timestamp
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    return str(filename)


def load_saved_tests() -> List[Dict]:
    """載入已保存的測試"""
    save_dir = Path("results/saved_tests")
    if not save_dir.exists():
        return []
    
    tests = []
    for f in save_dir.glob("*.pkl"):
        try:
            with open(f, 'rb') as file:
                data = pickle.load(file)
                data['filepath'] = str(f)
                tests.append(data)
        except Exception:
            pass
    
    return sorted(tests, key=lambda x: x.get('timestamp', ''), reverse=True)


def delete_test(filepath: str) -> bool:
    """刪除測試結果"""
    try:
        Path(filepath).unlink()
        return True
    except Exception:
        return False
