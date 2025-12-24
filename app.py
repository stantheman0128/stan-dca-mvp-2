# app.py
"""
定期定額策略回測工具 - Streamlit 主應用程式
DCA Strategy Backtesting Tool - Full Featured Version
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, Any, List
import pickle
import os
import io

# 設定頁面必須在最前面
st.set_page_config(
    page_title="DCA 策略回測工具",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 載入自定義模組
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
from utils.grid_search import GridSearchOptimizer, GridSearchConfig


# ===================== CSS 樣式 =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;600;700&display=swap');

/* 只對特定文字元素套用中文字體 */
.stMarkdown, .stMarkdown p, .stMarkdown li,
h1, h2, h3, h4, h5, h6,
.stDataFrame, .stTable,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
.element-container p {
    font-family: "Noto Sans TC", "Microsoft JhengHei", sans-serif;
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
    padding: 20px;
    color: white;
    text-align: center;
    margin: 5px;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
}

.metric-label {
    font-size: 0.9rem;
    opacity: 0.9;
}

.positive { color: #4CAF50; }
.negative { color: #F44336; }

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    padding: 10px 20px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)


# ===================== 初始化 =====================
# 使用 cache_resource 保持單例，但這些物件本身很小
@st.cache_resource
def get_data_loader():
    return DataLoader()

@st.cache_resource
def get_backtest_engine():
    return BacktestEngine()

@st.cache_resource
def get_visualizer():
    return Visualizer()

@st.cache_resource
def get_report_generator():
    return ReportGenerator()

@st.cache_resource
def get_robustness_tester():
    return RobustnessTester()


# ===================== 記憶體管理 =====================
def clear_memory_cache():
    """清除記憶體緩存"""
    import gc
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
    import psutil
    import os
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


# ===================== 輔助函數 =====================
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


def render_metric_card(label: str, value: str, is_positive: bool = True):
    """渲染指標卡片"""
    color_class = "positive" if is_positive else "negative"
    st.markdown(f"""
    <div style="background: #f8f9fa; border-radius: 10px; padding: 15px; text-align: center; border-left: 4px solid {'#4CAF50' if is_positive else '#F44336'};">
        <div style="font-size: 0.85rem; color: #666; margin-bottom: 5px;">{label}</div>
        <div style="font-size: 1.5rem; font-weight: 700;" class="{color_class}">{value}</div>
    </div>
    """, unsafe_allow_html=True)


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


def save_test_result(results: Dict, config: Dict, name: str):
    """保存測試結果"""
    save_dir = "results/saved_tests"
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_dir}/test_{timestamp}_{name}.pkl"
    
    data = {
        'results': results,
        'config': config,
        'name': name,
        'timestamp': timestamp
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    return filename


def load_saved_tests():
    """載入已保存的測試"""
    save_dir = "results/saved_tests"
    if not os.path.exists(save_dir):
        return []
    
    tests = []
    for f in os.listdir(save_dir):
        if f.endswith('.pkl'):
            filepath = os.path.join(save_dir, f)
            try:
                with open(filepath, 'rb') as file:
                    data = pickle.load(file)
                    data['filepath'] = filepath
                    tests.append(data)
            except:
                pass
    
    return sorted(tests, key=lambda x: x.get('timestamp', ''), reverse=True)


# ===================== 主應用 =====================
def main():
    # 標題
    st.title("📈 定期定額策略回測工具")
    st.caption("DCA Strategy Backtesting Tool - 研究不同定期定額策略的歷史表現")
    
    # 側邊欄底部：記憶體管理
    with st.sidebar:
        st.divider()
        with st.expander("🔧 系統管理", expanded=False):
            try:
                mem_usage = get_memory_usage()
                st.metric("記憶體使用", f"{mem_usage:.1f} MB")
            except ImportError:
                st.info("安裝 psutil 可監控記憶體")
            
            if st.button("🗑️ 清除緩存", use_container_width=True):
                clear_memory_cache()
                st.success("✅ 緩存已清除")
                st.rerun()
    
    # 頁面選擇
    page = st.sidebar.radio(
        "📌 功能選擇",
        ["🏠 基本回測", "🔬 穩健性測試", "🌍 跨市場分析", "🎯 參數優化", "📁 測試管理"],
        index=0
    )
    
    if page == "🏠 基本回測":
        basic_backtest_page()
    elif page == "🔬 穩健性測試":
        robustness_test_page()
    elif page == "🌍 跨市場分析":
        cross_market_page()
    elif page == "🎯 參數優化":
        grid_search_page()
    elif page == "📁 測試管理":
        test_management_page()


def basic_backtest_page():
    """基本回測頁面"""
    data_loader = get_data_loader()
    
    # 側邊欄配置
    with st.sidebar:
        st.header("⚙️ 回測配置")
        
        # 市場選擇
        data_loader = get_data_loader()
        markets = data_loader.get_available_markets()
        market_options = {f"{v} ({k})": k for k, v in markets.items()}
        
        selected_market = st.selectbox(
            "選擇市場",
            options=list(market_options.keys()),
            index=0
        )
        symbol = market_options[selected_market]
        
        # 日期範圍
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "開始日期",
                value=date(2010, 1, 1),
                min_value=date(2005, 1, 1),
                max_value=date.today()
            )
        with col2:
            end_date = st.date_input(
                "結束日期",
                value=date.today(),
                min_value=date(2005, 1, 1),
                max_value=date.today()
            )
        
        # 投資頻率
        frequency = st.selectbox(
            "投資頻率",
            options=['M', 'W', 'Q'],
            format_func=lambda x: {'M': '每月', 'W': '每週', 'Q': '每季'}[x],
            index=0
        )
        
        # 每期投入金額
        base_amount = st.number_input(
            "每期基礎投入金額",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000
        )
        
        st.divider()
        
        # 策略選擇
        st.subheader("📊 選擇策略")
        selected_strategies = st.multiselect(
            "選擇要比較的策略",
            options=list(STRATEGY_REGISTRY.keys()),
            default=['V0: 純定期定額', 'V1: 跌深加碼']
        )
        
        # 策略參數（可展開）
        strategy_params = {}
        with st.expander("🔧 策略參數設定", expanded=False):
            if 'V1: 跌深加碼' in selected_strategies:
                st.markdown("**V1: 跌深加碼**")
                strategy_params['V1'] = {
                    'lookback_period': st.slider("高點回顧期間(天)", 60, 504, 252),
                    'dip_threshold_1': st.slider("第一級跌幅閾值", 0.05, 0.20, 0.10),
                    'multiplier_1': st.slider("第一級加碼倍數", 1.0, 3.0, 1.5),
                    'dip_threshold_2': st.slider("第二級跌幅閾值", 0.15, 0.40, 0.20),
                    'multiplier_2': st.slider("第二級加碼倍數", 1.5, 4.0, 2.0),
                }
            
            if 'V2: 趨勢過濾' in selected_strategies:
                st.markdown("**V2: 趨勢過濾**")
                strategy_params['V2'] = {
                    'ma_period': st.slider("移動平均週期", 50, 400, 200),
                    'ma_type': st.selectbox("MA 類型", ['SMA', 'EMA']),
                    'below_multiplier': st.slider("低於均線加碼倍數", 1.0, 3.0, 1.5),
                }
            
            if 'V3: 波動率調整' in selected_strategies:
                st.markdown("**V3: 波動率調整**")
                strategy_params['V3'] = {
                    'volatility_window': st.slider("波動率窗口(天)", 10, 60, 20),
                    'high_vol_threshold': st.slider("高波動閾值(倍)", 1.0, 3.0, 1.5),
                    'low_vol_threshold': st.slider("低波動閾值(倍)", 0.3, 1.0, 0.8),
                    'high_vol_multiplier': st.slider("高波動加碼倍數", 1.0, 3.0, 1.5),
                    'low_vol_multiplier': st.slider("低波動減碼倍數", 0.3, 1.0, 0.8),
                }
        
        st.divider()
        
        # 執行按鈕
        run_backtest = st.button("🚀 開始回測", type="primary", use_container_width=True)
    
    # 主內容區
    if run_backtest:
        if not selected_strategies:
            st.error("請至少選擇一個策略")
            return
        
        # 下載數據
        with st.spinner(f"正在下載 {symbol} 數據..."):
            try:
                market_data = data_loader.download(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
            except Exception as e:
                st.error(f"數據下載失敗: {str(e)}")
                return
        
        # 執行回測
        engine = get_backtest_engine()
        results = {}
        
        progress_bar = st.progress(0, text="正在執行回測...")
        
        for i, strategy_name in enumerate(selected_strategies):
            # 獲取策略參數
            param_key = strategy_name.split(':')[0]
            params = strategy_params.get(param_key, {})
            
            # 創建策略實例
            strategy = create_strategy_instance(strategy_name, base_amount, params)
            
            try:
                result = engine.run(
                    strategy=strategy,
                    market_data=market_data,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    frequency=frequency
                )
                results[strategy_name] = result
            except Exception as e:
                st.warning(f"策略 {strategy_name} 回測失敗: {str(e)}")
            
            progress_bar.progress((i + 1) / len(selected_strategies), text=f"已完成 {strategy_name}")
        
        progress_bar.empty()
        
        if not results:
            st.error("所有策略回測失敗")
            return
        
        # 存儲結果到 session state
        st.session_state['backtest_results'] = results
        st.session_state['symbol'] = symbol
        
        # 顯示結果
        display_results(results, symbol)
    
    elif 'backtest_results' in st.session_state:
        # 顯示之前的結果
        display_results(
            st.session_state['backtest_results'],
            st.session_state.get('symbol', 'Unknown')
        )
    else:
        # 首頁說明
        st.info("👈 請在左側配置回測參數，然後點擊「開始回測」")
        
        with st.expander("📖 工具說明", expanded=True):
            st.markdown("""
            ### 功能特點
            - **多策略對比**: 同時比較多種 DCA 策略變化
            - **20年回測**: 支援 2005-2025 年歷史數據
            - **跨市場**: 美股、台股、國際市場
            - **完整指標**: 報酬率、風險、夏普比率等
            - **互動圖表**: Plotly 互動式視覺化
            - **報告導出**: Excel 報告下載
            
            ### 支援策略
            | 策略 | 說明 |
            |------|------|
            | V0: 純定期定額 | 固定金額投入，作為基準 |
            | V1: 跌深加碼 | 價格下跌時增加投入 |
            | V2: 趨勢過濾 | 價格低於均線時加碼 |
            | V3: 波動率調整 | 高波動時加碼 |
            
            ### 使用步驟
            1. 選擇市場和時間範圍
            2. 選擇要比較的策略
            3. 調整策略參數（可選）
            4. 點擊「開始回測」
            5. 查看結果和圖表
            6. 下載報告
            """)


def display_results(results: Dict[str, Any], symbol: str):
    """顯示回測結果"""
    visualizer = get_visualizer()
    report_gen = get_report_generator()
    stats_calc = StatisticsCalculator()
    data_loader = get_data_loader()
    
    # 標籤頁
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 績效總覽", "📈 圖表分析", "📋 詳細數據", "🔬 統計檢驗", "📊 vs Buy&Hold", "📥 導出報告"
    ])
    
    with tab1:
        st.subheader(f"📊 績效總覽 - {symbol}")
        
        # 關鍵指標卡片
        for name, result in results.items():
            st.markdown(f"### {name}")
            m = result.metrics
            
            cols = st.columns(6)
            with cols[0]:
                is_pos = m.total_return >= 0
                render_metric_card("總報酬率", f"{'+' if is_pos else ''}{m.total_return:.2f}%", is_pos)
            with cols[1]:
                is_pos = m.cagr >= 0
                render_metric_card("年化報酬率", f"{'+' if is_pos else ''}{m.cagr:.2f}%", is_pos)
            with cols[2]:
                render_metric_card("最大回撤", f"{m.max_drawdown:.2f}%", False)
            with cols[3]:
                is_pos = m.sharpe_ratio >= 0
                render_metric_card("夏普比率", f"{m.sharpe_ratio:.2f}", is_pos)
            with cols[4]:
                render_metric_card("總投入", f"{m.total_invested:,.0f}", True)
            with cols[5]:
                is_pos = m.final_value >= m.total_invested
                render_metric_card("最終市值", f"{m.final_value:,.0f}", is_pos)
            
            st.divider()
        
        # 對比表格
        st.subheader("策略對比表")
        comparison_df = report_gen.generate_comparison_table(results)
        
        # 高亮最佳值
        def highlight_best(s):
            if s.name in ['總報酬率(%)', '年化報酬率(%)', '夏普比率', '索提諾比率', '卡爾馬比率', '勝率(%)', '最終市值']:
                is_max = s == s.max()
                return ['background-color: #C8E6C9' if v else '' for v in is_max]
            elif s.name in ['最大回撤(%)', '年化波動率(%)']:
                is_min = s == s.min()
                return ['background-color: #C8E6C9' if v else '' for v in is_min]
            return ['' for _ in s]
        
        styled_df = comparison_df.style.apply(highlight_best, axis=0).format("{:.2f}")
        st.dataframe(styled_df, use_container_width=True)
    
    with tab2:
        st.subheader("📈 圖表分析")
        
        # 權益曲線
        st.plotly_chart(
            visualizer.equity_curve_comparison(results, "權益曲線對比"),
            use_container_width=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 報酬率曲線
            st.plotly_chart(
                visualizer.return_curve(results, "累積報酬率曲線"),
                use_container_width=True
            )
        
        with col2:
            # 回撤曲線
            st.plotly_chart(
                visualizer.drawdown_curve(results, "回撤曲線"),
                use_container_width=True
            )
        
        # 績效指標對比
        st.plotly_chart(
            visualizer.metrics_comparison_bar(results, "績效指標對比"),
            use_container_width=True
        )
        
        # 風險-報酬散點圖
        st.plotly_chart(
            visualizer.risk_return_scatter(results, "風險-報酬分析"),
            use_container_width=True
        )
        
        # 年度報酬分析
        st.subheader("📅 年度報酬分析")
        import plotly.graph_objects as go
        
        # 計算每年報酬
        yearly_returns = {}
        for name, result in results.items():
            equity = result.equity_curve.copy()
            equity['year'] = pd.to_datetime(equity['date']).dt.year
            yearly = equity.groupby('year').apply(
                lambda x: (x['value'].iloc[-1] - x['value'].iloc[0]) / x['value'].iloc[0] * 100 if len(x) > 1 else 0
            )
            yearly_returns[name] = yearly
        
        # 創建年度報酬熱力圖
        years = sorted(set().union(*[set(y.index) for y in yearly_returns.values()]))
        yearly_df = pd.DataFrame(index=years)
        for name, yearly in yearly_returns.items():
            yearly_df[name] = yearly
        yearly_df = yearly_df.fillna(0)
        
        import plotly.express as px
        fig = px.imshow(
            yearly_df.T,
            labels=dict(x="年度", y="策略", color="報酬率(%)"),
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            aspect='auto',
            text_auto='.1f'
        )
        fig.update_layout(title='年度報酬率熱力圖')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📋 詳細數據")
        
        strategy_select = st.selectbox(
            "選擇策略查看交易記錄",
            options=list(results.keys())
        )
        
        if strategy_select:
            result = results[strategy_select]
            trans_df = result.to_transactions_df()
            
            st.markdown(f"**{strategy_select} - 交易記錄**")
            st.dataframe(
                trans_df.style.format({
                    '價格': '{:.2f}',
                    '投入金額': '{:,.0f}',
                    '買入股數': '{:.4f}',
                    '累積股數': '{:.4f}',
                    '累積成本': '{:,.0f}',
                    '市值': '{:,.0f}',
                    '報酬率(%)': '{:.2f}',
                    '投入倍數': '{:.2f}',
                }),
                use_container_width=True,
                height=400
            )
    
    with tab4:
        st.subheader("🔬 統計檢驗")
        
        if len(results) >= 2:
            # 準備報酬序列
            strategy_returns = {}
            for name, result in results.items():
                equity = result.equity_curve
                returns = equity['value'].pct_change().dropna()
                strategy_returns[name] = returns
            
            # 多策略比較
            st.markdown("### 策略兩兩比較（t 檢驗）")
            comparison_result = stats_calc.multiple_comparison(strategy_returns)
            st.dataframe(comparison_result, use_container_width=True)
            
            # 各策略統計摘要
            st.markdown("### 各策略報酬統計摘要")
            for name, returns in strategy_returns.items():
                with st.expander(f"{name} 統計摘要"):
                    summary = stats_calc.summary_statistics(returns)
                    col1, col2 = st.columns(2)
                    for i, (key, value) in enumerate(summary.items()):
                        with col1 if i % 2 == 0 else col2:
                            st.metric(key, f"{value}")
        else:
            st.info("請選擇至少兩個策略進行統計比較")
    
    with tab5:
        st.subheader(" vs Buy & Hold 對比")
        st.markdown("比較 DCA 策略與一次性投入的表現差異")
        
        # 取第一個策略的數據作為參考
        first_result = list(results.values())[0]
        total_invested = first_result.metrics.total_invested
        
        # 獲取市場數據計算 Buy & Hold
        try:
            equity_df = first_result.equity_curve
            start_date = pd.to_datetime(equity_df['date'].iloc[0]).date()
            end_date = pd.to_datetime(equity_df['date'].iloc[-1]).date()
            
            market_data = data_loader.download(symbol, start_date, end_date)
            bh_metrics = calculate_buy_and_hold(market_data, total_invested)
            
            # 對比表格
            comparison_data = []
            for name, result in results.items():
                m = result.metrics
                comparison_data.append({
                    '策略': name,
                    '總報酬率(%)': m.total_return,
                    '年化報酬率(%)': m.cagr,
                    '最大回撤(%)': m.max_drawdown,
                    '夏普比率': m.sharpe_ratio,
                    '最終市值': m.final_value
                })
            
            # 添加 Buy & Hold
            comparison_data.append({
                '策略': '📈 Buy & Hold',
                '總報酬率(%)': bh_metrics['total_return'],
                '年化報酬率(%)': bh_metrics['cagr'],
                '最大回撤(%)': bh_metrics['max_drawdown'],
                '夏普比率': bh_metrics['sharpe_ratio'],
                '最終市值': bh_metrics['final_value']
            })
            
            comp_df = pd.DataFrame(comparison_data)
            
            # 高亮 Buy & Hold
            def highlight_bh(row):
                if row['策略'] == '📈 Buy & Hold':
                    return ['background-color: #E3F2FD'] * len(row)
                return [''] * len(row)
            
            st.dataframe(comp_df.style.apply(highlight_bh, axis=1).format({
                '總報酬率(%)': '{:.2f}',
                '年化報酬率(%)': '{:.2f}',
                '最大回撤(%)': '{:.2f}',
                '夏普比率': '{:.2f}',
                '最終市值': '{:,.0f}'
            }), use_container_width=True)
            
            # 相對表現分析
            st.markdown("### 相對 Buy & Hold 表現")
            relative_data = []
            for name, result in results.items():
                m = result.metrics
                relative_data.append({
                    '策略': name,
                    '超額報酬(%)': m.total_return - bh_metrics['total_return'],
                    '回撤減少(%)': bh_metrics['max_drawdown'] - m.max_drawdown,  # 負值 = DCA 回撤更小
                    '夏普差異': m.sharpe_ratio - bh_metrics['sharpe_ratio']
                })
            
            rel_df = pd.DataFrame(relative_data)
            
            # 顏色標記正負
            def color_positive(val):
                if isinstance(val, (int, float)):
                    color = '#C8E6C9' if val > 0 else '#FFCDD2' if val < 0 else ''
                    return f'background-color: {color}'
                return ''
            
            st.dataframe(rel_df.style.applymap(color_positive, subset=['超額報酬(%)', '回撤減少(%)', '夏普差異']).format({
                '超額報酬(%)': '{:+.2f}',
                '回撤減少(%)': '{:+.2f}',
                '夏普差異': '{:+.2f}'
            }), use_container_width=True)
            
            # 視覺化對比
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # DCA 策略
            for name, result in results.items():
                equity = result.equity_curve
                returns = (equity['value'] - equity['cost']) / equity['cost'] * 100
                fig.add_trace(go.Scatter(
                    x=equity['date'],
                    y=returns,
                    name=name,
                    mode='lines'
                ))
            
            # Buy & Hold (優先使用 Close)
            price_col = 'Close' if 'Close' in market_data.columns else 'Adj Close'
            if price_col in market_data.columns:
                bh_prices = market_data[price_col]
                bh_returns = (bh_prices / bh_prices.iloc[0] - 1) * 100
                fig.add_trace(go.Scatter(
                    x=market_data.index,
                    y=bh_returns,
                    name='Buy & Hold',
                    mode='lines',
                    line=dict(dash='dash', color='gray', width=2)
                ))
            
            fig.add_hline(y=0, line_dash="dot", line_color="black", opacity=0.3)
            fig.update_layout(
                title='累積報酬率對比: DCA vs Buy & Hold',
                xaxis_title='日期',
                yaxis_title='報酬率 (%)',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 結論
            best_dca = max(results.items(), key=lambda x: x[1].metrics.sharpe_ratio)
            if best_dca[1].metrics.sharpe_ratio > bh_metrics['sharpe_ratio']:
                st.success(f"💡 **{best_dca[0]}** 的風險調整後報酬優於 Buy & Hold (夏普比率 {best_dca[1].metrics.sharpe_ratio:.2f} vs {bh_metrics['sharpe_ratio']:.2f})")
            else:
                st.info(f"📊 Buy & Hold 的夏普比率 ({bh_metrics['sharpe_ratio']:.2f}) 優於所選 DCA 策略，但 DCA 可能有更低的心理壓力和更好的風險分散。")
                
        except Exception as e:
            st.warning(f"無法計算 Buy & Hold 對比: {e}")
    
    with tab6:
        st.subheader("📥 導出報告")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📊 Excel 報告")
            st.markdown("包含：概述、詳細指標、交易記錄")
            
            try:
                excel_data = report_gen.generate_excel_report(results, return_bytes=True)
                st.download_button(
                    label="📥 下載 Excel 報告",
                    data=excel_data,
                    file_name=f"DCA_Report_{symbol}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except ImportError:
                st.warning("需要安裝 openpyxl 套件才能導出 Excel")
        
        with col2:
            st.markdown("### 📄 CSV 數據")
            st.markdown("下載對比表格數據")
            
            csv_data = report_gen.generate_comparison_table(results).to_csv()
            st.download_button(
                label="📥 下載 CSV 數據",
                data=csv_data,
                file_name=f"DCA_Comparison_{symbol}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            st.markdown("### 💾 儲存結果")
            st.markdown("儲存到本地以便日後比較")
            
            test_name = st.text_input(
                "測試名稱",
                value=f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}",
                key="save_test_name_tab6"
            )
            
            if st.button("💾 儲存回測結果", use_container_width=True, key="save_btn_tab6"):
                try:
                    # 儲存資料夾
                    save_dir = Path("saved_tests")
                    save_dir.mkdir(exist_ok=True)
                    
                    # 準備儲存資料
                    save_data = {
                        "timestamp": datetime.now().isoformat(),
                        "symbol": symbol,
                        "test_name": test_name,
                        "results": {}
                    }
                    
                    for name, result in results.items():
                        save_data["results"][name] = {
                            "metrics": {
                                "total_return": result.metrics.total_return,
                                "annualized_return": result.metrics.annualized_return,
                                "volatility": result.metrics.volatility,
                                "sharpe_ratio": result.metrics.sharpe_ratio,
                                "max_drawdown": result.metrics.max_drawdown,
                                "total_invested": result.metrics.total_invested,
                                "final_value": result.metrics.final_value,
                                "total_periods": result.metrics.total_periods,
                            },
                            "config": {
                                "strategy_type": result.config.strategy_type,
                                "invest_amount": result.config.invest_amount,
                                "frequency": result.config.frequency,
                            }
                        }
                    
                    # 儲存 JSON
                    file_path = save_dir / f"{test_name}.json"
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(save_data, f, ensure_ascii=False, indent=2)
                    
                    st.success(f"✅ 已儲存到: {file_path}")
                except Exception as e:
                    st.error(f"儲存失敗: {e}")


# ===================== 穩健性測試頁面 =====================
def robustness_test_page():
    """穩健性測試頁面"""
    st.header("🔬 穩健性測試")
    st.caption("測試策略在不同時間點和模擬情境下的表現穩定性")
    
    data_loader = get_data_loader()
    engine = get_backtest_engine()
    robustness_tester = get_robustness_tester()
    
    with st.sidebar:
        st.header("⚙️ 測試配置")
        
        # 市場選擇
        markets = data_loader.get_available_markets()
        market_options = {f"{v} ({k})": k for k, v in markets.items()}
        selected_market = st.selectbox("選擇市場", list(market_options.keys()), key="robust_market")
        symbol = market_options[selected_market]
        
        # 策略選擇
        strategy_name = st.selectbox(
            "選擇策略",
            options=list(STRATEGY_REGISTRY.keys()),
            key="robust_strategy"
        )
        
        # 基礎金額
        base_amount = st.number_input("每期投入金額", 1000, 100000, 10000, key="robust_amount")
        
        # 投資頻率
        frequency = st.selectbox(
            "投資頻率",
            ['M', 'W', 'Q'],
            format_func=lambda x: {'M': '每月', 'W': '每週', 'Q': '每季'}[x],
            key="robust_freq"
        )
    
    # 測試類型選擇
    test_type = st.radio(
        "選擇測試類型",
        ["📅 固定起始點測試", "🎲 Monte Carlo 模擬", "📊 滾動窗口分析", "🎛️ 參數敏感度分析"],
        horizontal=True
    )
    
    if test_type == "📅 固定起始點測試":
        st.subheader("固定起始點測試")
        st.markdown("在不同歷史時間點開始投資，觀察策略表現差異")
        
        # 預設測試點
        test_points = {
            "2005-12-23": "完整 20 年",
            "2008-01-01": "金融海嘯前",
            "2009-03-01": "海嘯後谷底",
            "2015-01-01": "中期起點",
            "2020-01-01": "疫情前",
            "2020-04-01": "疫情後反彈"
        }
        
        selected_points = st.multiselect(
            "選擇測試起始點",
            options=list(test_points.keys()),
            default=list(test_points.keys())[:4],
            format_func=lambda x: f"{x} ({test_points[x]})"
        )
        
        if st.button("🚀 執行固定起始點測試", type="primary"):
            if not selected_points:
                st.error("請選擇至少一個起始點")
                return
            
            with st.spinner("正在下載數據..."):
                market_data = data_loader.download(symbol, date(2005, 1, 1), date.today())
            
            strategy = create_strategy_instance(strategy_name, base_amount, {})
            
            results_list = []
            progress = st.progress(0)
            
            for i, start_str in enumerate(selected_points):
                start_dt = datetime.strptime(start_str, "%Y-%m-%d").date()
                try:
                    result = engine.run(
                        strategy=strategy,
                        market_data=market_data,
                        symbol=symbol,
                        start_date=start_dt,
                        end_date=date.today(),
                        frequency=frequency
                    )
                    results_list.append({
                        '起始日期': start_str,
                        '說明': test_points[start_str],
                        '總報酬率(%)': result.metrics.total_return,
                        '年化報酬率(%)': result.metrics.cagr,
                        '最大回撤(%)': result.metrics.max_drawdown,
                        '夏普比率': result.metrics.sharpe_ratio,
                        '投資期間(年)': result.metrics.investment_years
                    })
                except Exception as e:
                    st.warning(f"起始點 {start_str} 測試失敗: {e}")
                
                progress.progress((i + 1) / len(selected_points))
            
            progress.empty()
            
            if results_list:
                df = pd.DataFrame(results_list)
                st.dataframe(df.style.format({
                    '總報酬率(%)': '{:.2f}',
                    '年化報酬率(%)': '{:.2f}',
                    '最大回撤(%)': '{:.2f}',
                    '夏普比率': '{:.2f}',
                    '投資期間(年)': '{:.1f}'
                }), use_container_width=True)
                
                # 視覺化
                import plotly.express as px
                fig = px.bar(df, x='起始日期', y='總報酬率(%)', 
                           title=f'{strategy_name} - 不同起始點總報酬率',
                           color='總報酬率(%)',
                           color_continuous_scale='RdYlGn')
                st.plotly_chart(fig, use_container_width=True)
    
    elif test_type == "🎲 Monte Carlo 模擬":
        st.subheader("Monte Carlo 隨機模擬")
        st.markdown("隨機選擇起始日期和投資期間，進行大量模擬測試")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            num_simulations = st.slider("模擬次數", 50, 500, 200)
        with col2:
            min_years = st.slider("最短投資年數", 1, 5, 3)
        with col3:
            max_years = st.slider("最長投資年數", 5, 20, 15)
        
        if st.button("🎲 執行 Monte Carlo 模擬", type="primary"):
            with st.spinner("正在下載數據..."):
                market_data = data_loader.download(symbol, date(2005, 1, 1), date.today())
            
            strategy = create_strategy_instance(strategy_name, base_amount, {})
            
            progress_bar = st.progress(0, text="執行模擬中...")
            status_text = st.empty()
            
            results = []
            for i in range(num_simulations):
                # 隨機選擇起始日期和投資期間
                available_days = len(market_data)
                min_days = min_years * 252
                max_days = min(max_years * 252, available_days - 252)
                
                if max_days <= min_days:
                    continue
                
                duration_days = np.random.randint(min_days, max_days)
                max_start_idx = available_days - duration_days
                start_idx = np.random.randint(0, max_start_idx)
                
                start_date_sim = market_data.index[start_idx].date()
                end_date_sim = market_data.index[start_idx + duration_days].date()
                
                try:
                    result = engine.run(
                        strategy=strategy,
                        market_data=market_data,
                        symbol=symbol,
                        start_date=start_date_sim,
                        end_date=end_date_sim,
                        frequency=frequency
                    )
                    results.append({
                        'total_return': result.metrics.total_return,
                        'cagr': result.metrics.cagr,
                        'max_drawdown': result.metrics.max_drawdown,
                        'sharpe_ratio': result.metrics.sharpe_ratio,
                        'years': result.metrics.investment_years
                    })
                except:
                    pass
                
                progress_bar.progress((i + 1) / num_simulations)
                status_text.text(f"已完成 {i+1}/{num_simulations} 次模擬")
            
            progress_bar.empty()
            status_text.empty()
            
            if results:
                df = pd.DataFrame(results)
                
                # 統計摘要
                st.markdown("### 📊 統計摘要")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("平均報酬率", f"{df['total_return'].mean():.2f}%")
                with col2:
                    st.metric("中位數報酬率", f"{df['total_return'].median():.2f}%")
                with col3:
                    st.metric("勝率 (>0%)", f"{(df['total_return'] > 0).mean() * 100:.1f}%")
                with col4:
                    st.metric("95% CI", f"[{df['total_return'].quantile(0.025):.1f}%, {df['total_return'].quantile(0.975):.1f}%]")
                
                # 分佈圖
                import plotly.express as px
                import plotly.graph_objects as go
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=df['total_return'], nbinsx=30, name='報酬率分佈'))
                fig.add_vline(x=df['total_return'].mean(), line_dash="dash", line_color="red",
                            annotation_text=f"平均: {df['total_return'].mean():.1f}%")
                fig.add_vline(x=0, line_dash="dot", line_color="gray")
                fig.update_layout(
                    title=f'{strategy_name} - Monte Carlo 報酬率分佈 ({num_simulations} 次模擬)',
                    xaxis_title='總報酬率 (%)',
                    yaxis_title='次數'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 詳細統計
                st.markdown("### 詳細統計")
                stats_df = pd.DataFrame({
                    '指標': ['平均', '標準差', '最小', '25%', '中位數', '75%', '最大'],
                    '總報酬率(%)': [
                        df['total_return'].mean(),
                        df['total_return'].std(),
                        df['total_return'].min(),
                        df['total_return'].quantile(0.25),
                        df['total_return'].median(),
                        df['total_return'].quantile(0.75),
                        df['total_return'].max()
                    ],
                    '年化報酬率(%)': [
                        df['cagr'].mean(),
                        df['cagr'].std(),
                        df['cagr'].min(),
                        df['cagr'].quantile(0.25),
                        df['cagr'].median(),
                        df['cagr'].quantile(0.75),
                        df['cagr'].max()
                    ],
                    '夏普比率': [
                        df['sharpe_ratio'].mean(),
                        df['sharpe_ratio'].std(),
                        df['sharpe_ratio'].min(),
                        df['sharpe_ratio'].quantile(0.25),
                        df['sharpe_ratio'].median(),
                        df['sharpe_ratio'].quantile(0.75),
                        df['sharpe_ratio'].max()
                    ]
                })
                st.dataframe(stats_df.style.format({
                    '總報酬率(%)': '{:.2f}',
                    '年化報酬率(%)': '{:.2f}',
                    '夏普比率': '{:.2f}'
                }), use_container_width=True)
    
    elif test_type == "📊 滾動窗口分析":
        st.subheader("滾動窗口分析")
        st.markdown("使用固定時間窗口在歷史上滾動，觀察策略穩定性")
        
        col1, col2 = st.columns(2)
        with col1:
            window_years = st.selectbox("窗口大小 (年)", [1, 3, 5, 10], index=1)
        with col2:
            step_months = st.slider("滾動步長 (月)", 1, 12, 3)
        
        if st.button("📊 執行滾動窗口分析", type="primary"):
            with st.spinner("正在下載數據..."):
                market_data = data_loader.download(symbol, date(2005, 1, 1), date.today())
            
            strategy = create_strategy_instance(strategy_name, base_amount, {})
            
            window_days = window_years * 365
            step_days = step_months * 30
            
            results = []
            start_idx = 0
            total_windows = (len(market_data) - window_days) // step_days
            progress = st.progress(0)
            
            window_count = 0
            while start_idx + window_days < len(market_data):
                start_dt = market_data.index[start_idx].date()
                end_idx = min(start_idx + window_days, len(market_data) - 1)
                end_dt = market_data.index[end_idx].date()
                
                try:
                    result = engine.run(
                        strategy=strategy,
                        market_data=market_data,
                        symbol=symbol,
                        start_date=start_dt,
                        end_date=end_dt,
                        frequency=frequency
                    )
                    results.append({
                        '窗口起始': start_dt,
                        '窗口結束': end_dt,
                        '總報酬率(%)': result.metrics.total_return,
                        '年化報酬率(%)': result.metrics.cagr,
                        '夏普比率': result.metrics.sharpe_ratio
                    })
                except:
                    pass
                
                start_idx += step_days
                window_count += 1
                if total_windows > 0:
                    progress.progress(min(window_count / total_windows, 1.0))
            
            progress.empty()
            
            if results:
                df = pd.DataFrame(results)
                
                # 摘要統計
                st.markdown("### 統計摘要")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("平均報酬率", f"{df['總報酬率(%)'].mean():.2f}%")
                with col2:
                    st.metric("標準差", f"{df['總報酬率(%)'].std():.2f}%")
                with col3:
                    st.metric("最佳窗口", f"{df['總報酬率(%)'].max():.2f}%")
                with col4:
                    st.metric("最差窗口", f"{df['總報酬率(%)'].min():.2f}%")
                
                # 時間序列圖
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['窗口起始'],
                    y=df['總報酬率(%)'],
                    mode='lines+markers',
                    name='報酬率',
                    line=dict(color='blue')
                ))
                fig.add_hline(y=df['總報酬率(%)'].mean(), line_dash="dash", line_color="red",
                            annotation_text=f"平均: {df['總報酬率(%)'].mean():.1f}%")
                fig.add_hline(y=0, line_dash="dot", line_color="gray")
                fig.update_layout(
                    title=f'{strategy_name} - {window_years}年滾動窗口報酬率',
                    xaxis_title='窗口起始日期',
                    yaxis_title='總報酬率 (%)'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 詳細數據
                st.dataframe(df.style.format({
                    '總報酬率(%)': '{:.2f}',
                    '年化報酬率(%)': '{:.2f}',
                    '夏普比率': '{:.2f}'
                }), use_container_width=True, height=300)
    
    elif test_type == "🎛️ 參數敏感度分析":
        st.subheader("參數敏感度分析")
        st.markdown("測試策略參數變化對結果的影響")
        
        # 只對 V1 跌深加碼策略進行參數敏感度分析
        if strategy_name == 'V1: 跌深加碼':
            param_to_test = st.selectbox(
                "選擇要測試的參數",
                ["dip_threshold_1 (跌幅閾值)", "multiplier_1 (加碼倍數)", "lookback_period (回顧期間)"]
            )
            
            if "dip_threshold_1" in param_to_test:
                param_values = [0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.25]
                param_key = 'dip_threshold_1'
            elif "multiplier_1" in param_to_test:
                param_values = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
                param_key = 'multiplier_1'
            else:
                param_values = [60, 126, 189, 252, 378, 504]
                param_key = 'lookback_period'
            
            col1, col2 = st.columns(2)
            with col1:
                start_date_sens = st.date_input("開始日期", date(2010, 1, 1), key="sens_start")
            with col2:
                end_date_sens = st.date_input("結束日期", date.today(), key="sens_end")
            
            if st.button("🎛️ 執行參數敏感度分析", type="primary"):
                with st.spinner("正在下載數據..."):
                    market_data = data_loader.download(symbol, start_date_sens, end_date_sens)
                
                results = []
                progress = st.progress(0)
                
                for i, param_val in enumerate(param_values):
                    params = {
                        'lookback_period': 252,
                        'dip_threshold_1': 0.10,
                        'multiplier_1': 1.5,
                        'dip_threshold_2': 0.20,
                        'multiplier_2': 2.0
                    }
                    params[param_key] = param_val
                    
                    strategy = DCADipBuyingStrategy(
                        base_amount=base_amount,
                        **params
                    )
                    
                    try:
                        result = engine.run(
                            strategy=strategy,
                            market_data=market_data,
                            symbol=symbol,
                            start_date=start_date_sens,
                            end_date=end_date_sens,
                            frequency=frequency
                        )
                        results.append({
                            '參數值': param_val,
                            '總報酬率(%)': result.metrics.total_return,
                            '年化報酬率(%)': result.metrics.cagr,
                            '夏普比率': result.metrics.sharpe_ratio,
                            '最大回撤(%)': result.metrics.max_drawdown
                        })
                    except Exception as e:
                        st.warning(f"參數 {param_val} 測試失敗: {e}")
                    
                    progress.progress((i + 1) / len(param_values))
                
                progress.empty()
                
                if results:
                    df = pd.DataFrame(results)
                    
                    # 顯示表格
                    st.dataframe(df.style.format({
                        '總報酬率(%)': '{:.2f}',
                        '年化報酬率(%)': '{:.2f}',
                        '夏普比率': '{:.2f}',
                        '最大回撤(%)': '{:.2f}'
                    }), use_container_width=True)
                    
                    # 視覺化
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots
                    
                    fig = make_subplots(rows=2, cols=2, 
                                       subplot_titles=('總報酬率', '年化報酬率', '夏普比率', '最大回撤'))
                    
                    fig.add_trace(go.Scatter(x=df['參數值'], y=df['總報酬率(%)'], mode='lines+markers', name='總報酬率'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df['參數值'], y=df['年化報酬率(%)'], mode='lines+markers', name='年化報酬率'), row=1, col=2)
                    fig.add_trace(go.Scatter(x=df['參數值'], y=df['夏普比率'], mode='lines+markers', name='夏普比率'), row=2, col=1)
                    fig.add_trace(go.Scatter(x=df['參數值'], y=df['最大回撤(%)'], mode='lines+markers', name='最大回撤'), row=2, col=2)
                    
                    fig.update_layout(title=f'V1 跌深加碼 - {param_to_test} 敏感度分析', showlegend=False, height=600)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 最佳參數建議
                    best_sharpe_idx = df['夏普比率'].idxmax()
                    st.success(f"💡 最佳夏普比率參數: {df.loc[best_sharpe_idx, '參數值']} (夏普比率: {df.loc[best_sharpe_idx, '夏普比率']:.2f})")
        
        elif strategy_name == 'V2: 趨勢過濾':
            param_to_test = st.selectbox(
                "選擇要測試的參數",
                ["ma_period (均線週期)", "below_multiplier (加碼倍數)"]
            )
            
            if "ma_period" in param_to_test:
                param_values = [50, 100, 150, 200, 250, 300, 400]
                param_key = 'ma_period'
            else:
                param_values = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
                param_key = 'below_multiplier'
            
            col1, col2 = st.columns(2)
            with col1:
                start_date_sens = st.date_input("開始日期", date(2010, 1, 1), key="sens_start_v2")
            with col2:
                end_date_sens = st.date_input("結束日期", date.today(), key="sens_end_v2")
            
            if st.button("🎛️ 執行參數敏感度分析", type="primary", key="sens_btn_v2"):
                with st.spinner("正在下載數據..."):
                    market_data = data_loader.download(symbol, start_date_sens, end_date_sens)
                
                results = []
                progress = st.progress(0)
                
                for i, param_val in enumerate(param_values):
                    params = {
                        'ma_period': 200,
                        'ma_type': 'SMA',
                        'below_multiplier': 1.5
                    }
                    params[param_key] = param_val
                    
                    strategy = DCATrendFilterStrategy(
                        base_amount=base_amount,
                        **params
                    )
                    
                    try:
                        result = engine.run(
                            strategy=strategy,
                            market_data=market_data,
                            symbol=symbol,
                            start_date=start_date_sens,
                            end_date=end_date_sens,
                            frequency=frequency
                        )
                        results.append({
                            '參數值': param_val,
                            '總報酬率(%)': result.metrics.total_return,
                            '年化報酬率(%)': result.metrics.cagr,
                            '夏普比率': result.metrics.sharpe_ratio,
                            '最大回撤(%)': result.metrics.max_drawdown
                        })
                    except Exception as e:
                        st.warning(f"參數 {param_val} 測試失敗: {e}")
                    
                    progress.progress((i + 1) / len(param_values))
                
                progress.empty()
                
                if results:
                    df = pd.DataFrame(results)
                    st.dataframe(df.style.format({
                        '總報酬率(%)': '{:.2f}',
                        '年化報酬率(%)': '{:.2f}',
                        '夏普比率': '{:.2f}',
                        '最大回撤(%)': '{:.2f}'
                    }), use_container_width=True)
                    
                    import plotly.express as px
                    fig = px.line(df, x='參數值', y=['總報酬率(%)', '夏普比率'], 
                                 title=f'V2 趨勢過濾 - {param_to_test} 敏感度分析')
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("請選擇 V1 跌深加碼 或 V2 趨勢過濾 策略進行參數敏感度分析")


# ===================== 跨市場分析頁面 =====================
def cross_market_page():
    """跨市場分析頁面"""
    st.header("🌍 跨市場分析")
    st.caption("比較策略在不同市場的表現")
    
    data_loader = get_data_loader()
    engine = get_backtest_engine()
    
    with st.sidebar:
        st.header("⚙️ 配置")
        
        # 日期範圍
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("開始日期", date(2010, 1, 1), key="cross_start")
        with col2:
            end_date = st.date_input("結束日期", date.today(), key="cross_end")
        
        # 基礎金額
        base_amount = st.number_input("每期投入金額", 1000, 100000, 10000, key="cross_amount")
        
        # 投資頻率
        frequency = st.selectbox(
            "投資頻率",
            ['M', 'W', 'Q'],
            format_func=lambda x: {'M': '每月', 'W': '每週', 'Q': '每季'}[x],
            key="cross_freq"
        )
    
    # 市場選擇
    markets = data_loader.get_available_markets()
    selected_markets = st.multiselect(
        "選擇要測試的市場",
        options=list(markets.keys()),
        default=['SPY', 'QQQ', '0050.TW'],
        format_func=lambda x: f"{markets[x]} ({x})"
    )
    
    # 策略選擇
    selected_strategies = st.multiselect(
        "選擇要比較的策略",
        options=list(STRATEGY_REGISTRY.keys()),
        default=['V0: 純定期定額', 'V1: 跌深加碼']
    )
    
    if st.button("🌍 執行跨市場分析", type="primary"):
        if not selected_markets or not selected_strategies:
            st.error("請選擇至少一個市場和一個策略")
            return
        
        all_results = {}
        total_tests = len(selected_markets) * len(selected_strategies)
        progress = st.progress(0)
        current = 0
        
        for market_symbol in selected_markets:
            with st.spinner(f"下載 {market_symbol} 數據..."):
                try:
                    market_data = data_loader.download(market_symbol, start_date, end_date)
                except Exception as e:
                    st.warning(f"無法下載 {market_symbol}: {e}")
                    current += len(selected_strategies)
                    continue
            
            for strategy_name in selected_strategies:
                strategy = create_strategy_instance(strategy_name, base_amount, {})
                
                try:
                    result = engine.run(
                        strategy=strategy,
                        market_data=market_data,
                        symbol=market_symbol,
                        start_date=start_date,
                        end_date=end_date,
                        frequency=frequency
                    )
                    key = f"{market_symbol}_{strategy_name}"
                    all_results[key] = {
                        'market': market_symbol,
                        'market_name': markets[market_symbol],
                        'strategy': strategy_name,
                        'metrics': result.metrics
                    }
                except Exception as e:
                    st.warning(f"{market_symbol} - {strategy_name} 失敗: {e}")
                
                current += 1
                progress.progress(current / total_tests)
        
        progress.empty()
        
        if all_results:
            # 構建結果表格
            table_data = []
            for key, data in all_results.items():
                m = data['metrics']
                table_data.append({
                    '市場': data['market'],
                    '市場名稱': data['market_name'],
                    '策略': data['strategy'],
                    '總報酬率(%)': m.total_return,
                    '年化報酬率(%)': m.cagr,
                    '最大回撤(%)': m.max_drawdown,
                    '夏普比率': m.sharpe_ratio,
                    '索提諾比率': m.sortino_ratio
                })
            
            df = pd.DataFrame(table_data)
            
            # 總覽表格
            st.subheader("📊 跨市場結果總覽")
            st.dataframe(df.style.format({
                '總報酬率(%)': '{:.2f}',
                '年化報酬率(%)': '{:.2f}',
                '最大回撤(%)': '{:.2f}',
                '夏普比率': '{:.2f}',
                '索提諾比率': '{:.2f}'
            }), use_container_width=True)
            
            # 熱力圖
            st.subheader("📈 策略-市場報酬率熱力圖")
            pivot_df = df.pivot(index='策略', columns='市場', values='總報酬率(%)')
            
            import plotly.express as px
            fig = px.imshow(
                pivot_df,
                labels=dict(x="市場", y="策略", color="總報酬率(%)"),
                color_continuous_scale='RdYlGn',
                aspect='auto',
                text_auto='.1f'
            )
            fig.update_layout(title='策略 × 市場 報酬率熱力圖')
            st.plotly_chart(fig, use_container_width=True)
            
            # 策略穩健性評分
            st.subheader("🏆 策略穩健性評分")
            st.markdown("標準差越小 = 跨市場表現越穩定")
            
            stability_data = []
            for strategy in selected_strategies:
                strategy_df = df[df['策略'] == strategy]
                if len(strategy_df) > 1:
                    stability_data.append({
                        '策略': strategy,
                        '平均報酬率(%)': strategy_df['總報酬率(%)'].mean(),
                        '報酬率標準差(%)': strategy_df['總報酬率(%)'].std(),
                        '最佳市場': strategy_df.loc[strategy_df['總報酬率(%)'].idxmax(), '市場'],
                        '最差市場': strategy_df.loc[strategy_df['總報酬率(%)'].idxmin(), '市場']
                    })
            
            if stability_data:
                stability_df = pd.DataFrame(stability_data)
                st.dataframe(stability_df.style.format({
                    '平均報酬率(%)': '{:.2f}',
                    '報酬率標準差(%)': '{:.2f}'
                }), use_container_width=True)


# ===================== 參數優化頁面 =====================
def grid_search_page():
    """參數優化（網格搜索）頁面"""
    st.header("🎯 參數優化 - 全自動網格搜索")
    st.caption("自動測試所有策略、市場、參數組合，找出最佳配置")
    
    data_loader = get_data_loader()
    markets = data_loader.get_available_markets()
    
    # 配置區域
    with st.sidebar:
        st.header("⚙️ 搜索配置")
        
        # 模式選擇
        search_mode = st.radio(
            "搜索模式",
            ["⚡ 快速模式", "🔬 完整模式", "🛠️ 自定義"],
            help="快速模式：約 8 種組合\n完整模式：約 500+ 種組合\n自定義：自行設定"
        )
        
        st.divider()
        
        # 策略選擇
        st.subheader("📊 策略選擇")
        selected_strategies = []
        if st.checkbox("V0: 純定期定額", value=True):
            selected_strategies.append('V0')
        if st.checkbox("V1: 跌深加碼", value=True):
            selected_strategies.append('V1')
        if st.checkbox("V2: 趨勢過濾", value=True):
            selected_strategies.append('V2')
        if st.checkbox("V3: 波動率調整", value=True):
            selected_strategies.append('V3')
        
        st.divider()
        
        # 市場選擇
        st.subheader("🌍 市場選擇")
        market_options = list(markets.keys())
        # 確保預設值都在選項中
        default_fast = [m for m in ['SPY', '0050.TW'] if m in market_options]
        default_full = [m for m in ['SPY', 'QQQ', '0050.TW', 'DIA'] if m in market_options]
        selected_markets = st.multiselect(
            "選擇市場",
            market_options,
            default=default_fast if search_mode == "⚡ 快速模式" else default_full
        )
        
        st.divider()
        
        # 時間區間
        st.subheader("📅 時間區間")
        if search_mode == "🛠️ 自定義":
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("開始日期", value=date(2015, 1, 1))
            with col2:
                end_date = st.date_input("結束日期", value=date(2024, 12, 31))
            
            test_multiple_periods = st.checkbox("測試多個起始點", value=False)
            if test_multiple_periods:
                additional_starts = st.multiselect(
                    "額外起始年份",
                    [2010, 2012, 2016, 2018, 2020],
                    default=[2010, 2018]
                )
        else:
            start_date = date(2015, 1, 1)
            end_date = date(2024, 12, 31)
            test_multiple_periods = search_mode == "🔬 完整模式"
            additional_starts = [2010, 2018] if test_multiple_periods else []
    
    # 主要內容區
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 搜索範圍預覽")
        
        # 建立配置
        if search_mode == "⚡ 快速模式":
            config = GridSearchConfig(
                strategies=selected_strategies,
                markets=selected_markets,
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
        elif search_mode == "🔬 完整模式":
            config = GridSearchConfig(
                strategies=selected_strategies,
                markets=selected_markets,
                start_dates=[date(2010, 1, 1), date(2015, 1, 1), date(2018, 1, 1)],
                end_dates=[date(2024, 12, 31), date(2024, 12, 31), date(2024, 12, 31)],
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
        else:
            # 自定義模式
            start_dates = [start_date]
            end_dates = [end_date]
            if test_multiple_periods:
                for year in additional_starts:
                    start_dates.append(date(year, 1, 1))
                    end_dates.append(end_date)
            
            config = GridSearchConfig(
                strategies=selected_strategies,
                markets=selected_markets,
                start_dates=start_dates,
                end_dates=end_dates,
                base_amounts=[10000],
                frequencies=['monthly'],
            )
        
        # 計算組合數
        optimizer = GridSearchOptimizer()
        total_combinations = optimizer.count_total_combinations(config)
        
        # 顯示預覽
        preview_col1, preview_col2, preview_col3 = st.columns(3)
        with preview_col1:
            st.metric("策略數", len(selected_strategies))
        with preview_col2:
            st.metric("市場數", len(selected_markets))
        with preview_col3:
            st.metric("總組合數", f"{total_combinations:,}")
        
        estimated_time = total_combinations * 0.5  # 估計每組合 0.5 秒
        st.info(f"⏱️ 預估執行時間：約 {estimated_time:.0f} 秒 ({estimated_time/60:.1f} 分鐘)")
    
    with col2:
        st.subheader("🎮 執行控制")
        
        run_button = st.button(
            "🚀 開始搜索",
            type="primary",
            use_container_width=True,
            disabled=len(selected_strategies) == 0 or len(selected_markets) == 0
        )
    
    # 執行搜索
    if run_button:
        st.divider()
        st.subheader("📊 搜索進度")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total):
            progress = current / total
            progress_bar.progress(progress)
            status_text.text(f"進度：{current}/{total} ({progress*100:.1f}%)")
        
        with st.spinner("正在執行網格搜索..."):
            start_time = datetime.now()
            results_df = optimizer.run_grid_search(config, progress_callback=update_progress)
            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ 完成！共測試 {len(results_df)} 種組合，耗時 {elapsed:.1f} 秒")
        
        # 儲存結果到 session state
        st.session_state['grid_search_results'] = results_df
        st.session_state['grid_search_optimizer'] = optimizer
        st.rerun()
    
    # 顯示結果
    if 'grid_search_results' in st.session_state and len(st.session_state['grid_search_results']) > 0:
        results_df = st.session_state['grid_search_results']
        optimizer = st.session_state.get('grid_search_optimizer', GridSearchOptimizer())
        
        st.divider()
        st.subheader("🏆 搜索結果")
        
        # 結果標籤頁
        result_tabs = st.tabs([
            "📊 完整結果",
            "🥇 最佳 Sharpe",
            "💰 最佳報酬",
            "🛡️ 最低回撤",
            "📈 策略彙總",
            "🌍 市場彙總",
            "📉 熱力圖"
        ])
        
        with result_tabs[0]:
            st.caption(f"共 {len(results_df)} 種組合（依 Sharpe Ratio 排序）")
            
            # 篩選器
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            with filter_col1:
                filter_strategy = st.multiselect(
                    "篩選策略",
                    results_df['策略'].unique().tolist(),
                    default=results_df['策略'].unique().tolist()
                )
            with filter_col2:
                filter_market = st.multiselect(
                    "篩選市場",
                    results_df['市場'].unique().tolist(),
                    default=results_df['市場'].unique().tolist()
                )
            with filter_col3:
                min_sharpe = st.number_input("最低 Sharpe", value=0.0, step=0.1)
            
            filtered_df = results_df[
                (results_df['策略'].isin(filter_strategy)) &
                (results_df['市場'].isin(filter_market)) &
                (results_df['夏普比率'] >= min_sharpe)
            ]
            
            st.dataframe(
                filtered_df.style.format({
                    '總投入': '{:,.0f}',
                    '最終價值': '{:,.0f}',
                    '總報酬率(%)': '{:.2f}',
                    'CAGR(%)': '{:.2f}',
                    '夏普比率': '{:.3f}',
                    '最大回撤(%)': '{:.2f}',
                    '波動率(%)': '{:.2f}',
                    '勝率(%)': '{:.2f}',
                    '平均成本': '{:.2f}',
                }).background_gradient(subset=['夏普比率'], cmap='RdYlGn'),
                use_container_width=True,
                height=400
            )
        
        with result_tabs[1]:
            st.caption("🥇 Sharpe Ratio 最高的 20 種組合")
            best_sharpe = optimizer.get_best_by_metric(results_df, '夏普比率', 20)
            st.dataframe(
                best_sharpe.style.format({
                    '總報酬率(%)': '{:.2f}',
                    'CAGR(%)': '{:.2f}',
                    '夏普比率': '{:.3f}',
                    '最大回撤(%)': '{:.2f}',
                }).background_gradient(subset=['夏普比率'], cmap='RdYlGn'),
                use_container_width=True
            )
        
        with result_tabs[2]:
            st.caption("💰 總報酬率最高的 20 種組合")
            best_return = optimizer.get_best_by_metric(results_df, '總報酬率(%)', 20)
            st.dataframe(
                best_return.style.format({
                    '總報酬率(%)': '{:.2f}',
                    'CAGR(%)': '{:.2f}',
                    '夏普比率': '{:.3f}',
                    '最大回撤(%)': '{:.2f}',
                }).background_gradient(subset=['總報酬率(%)'], cmap='RdYlGn'),
                use_container_width=True
            )
        
        with result_tabs[3]:
            st.caption("🛡️ 最大回撤最低的 20 種組合")
            best_dd = optimizer.get_best_by_metric(results_df, '最大回撤(%)', 20)
            st.dataframe(
                best_dd.style.format({
                    '總報酬率(%)': '{:.2f}',
                    'CAGR(%)': '{:.2f}',
                    '夏普比率': '{:.3f}',
                    '最大回撤(%)': '{:.2f}',
                }).background_gradient(subset=['最大回撤(%)'], cmap='RdYlGn_r'),
                use_container_width=True
            )
        
        with result_tabs[4]:
            st.caption("📈 依策略分組的統計彙總")
            strategy_summary = optimizer.get_summary_by_strategy(results_df)
            st.dataframe(strategy_summary, use_container_width=True)
            
            # 策略比較圖表
            import plotly.express as px
            fig = px.box(
                results_df,
                x='策略',
                y='夏普比率',
                color='策略',
                title='各策略 Sharpe Ratio 分布'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with result_tabs[5]:
            st.caption("🌍 依市場分組的統計彙總")
            market_summary = optimizer.get_summary_by_market(results_df)
            st.dataframe(market_summary, use_container_width=True)
            
            # 市場比較圖表
            import plotly.express as px
            fig = px.box(
                results_df,
                x='市場',
                y='CAGR(%)',
                color='市場',
                title='各市場 CAGR 分布'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with result_tabs[6]:
            st.caption("📉 策略 x 市場 熱力圖")
            
            # 創建熱力圖數據
            metric_option = st.selectbox(
                "選擇指標",
                ['夏普比率', 'CAGR(%)', '總報酬率(%)', '最大回撤(%)']
            )
            
            pivot_df = results_df.pivot_table(
                values=metric_option,
                index='策略',
                columns='市場',
                aggfunc='mean'
            )
            
            import plotly.express as px
            fig = px.imshow(
                pivot_df,
                labels=dict(x="市場", y="策略", color=metric_option),
                x=pivot_df.columns.tolist(),
                y=pivot_df.index.tolist(),
                color_continuous_scale='RdYlGn' if metric_option != '最大回撤(%)' else 'RdYlGn_r',
                title=f'策略 x 市場：{metric_option} 熱力圖'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # 匯出按鈕
        st.divider()
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            excel_data = optimizer.export_to_excel(results_df, "grid_search_results.xlsx")
            st.download_button(
                label="📥 下載 Excel 報告",
                data=excel_data,
                file_name=f"DCA_GridSearch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col2:
            csv_data = results_df.to_csv(index=True).encode('utf-8-sig')
            st.download_button(
                label="📥 下載 CSV",
                data=csv_data,
                file_name=f"DCA_GridSearch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col3:
            if st.button("🗑️ 清除結果"):
                del st.session_state['grid_search_results']
                if 'grid_search_optimizer' in st.session_state:
                    del st.session_state['grid_search_optimizer']
                st.rerun()


# ===================== 測試管理頁面 =====================
def test_management_page():
    """測試管理頁面"""
    st.header("📁 測試管理")
    st.caption("保存、載入、匯出和管理測試結果")
    
    tab1, tab2, tab3 = st.tabs(["💾 保存測試", "📂 載入測試", "📥 快速匯出"])
    
    with tab1:
        st.subheader("保存當前測試結果")
        
        if 'backtest_results' in st.session_state and st.session_state['backtest_results']:
            results = st.session_state['backtest_results']
            symbol = st.session_state.get('symbol', 'Unknown')
            
            st.success(f"✅ 當前有 {len(results)} 個策略的測試結果 ({symbol})")
            
            # 顯示當前結果摘要
            with st.expander("📋 當前結果摘要", expanded=True):
                summary_data = []
                for name, result in results.items():
                    if hasattr(result, 'metrics'):
                        m = result.metrics
                        summary_data.append({
                            '策略': name,
                            '總報酬(%)': f"{m.total_return:.2f}",
                            'CAGR(%)': f"{m.cagr:.2f}",
                            'Sharpe': f"{m.sharpe_ratio:.3f}",
                            '最大回撤(%)': f"{m.max_drawdown:.2f}",
                        })
                if summary_data:
                    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                test_name = st.text_input(
                    "測試名稱", 
                    value=f"{symbol}_{datetime.now().strftime('%m%d')}",
                    help="輸入一個方便識別的名稱"
                )
            
            with col2:
                st.write("")  # 對齊
                st.write("")
                if st.button("💾 保存測試結果", type="primary", use_container_width=True):
                    config = {
                        'symbol': symbol,
                        'strategies': list(results.keys()),
                        'timestamp': datetime.now().isoformat(),
                        'start_date': str(st.session_state.get('start_date', '')),
                        'end_date': str(st.session_state.get('end_date', '')),
                    }
                    filepath = save_test_result(results, config, test_name)
                    st.success(f"✅ 測試已保存！")
                    st.caption(f"路徑: {filepath}")
        else:
            st.info("💡 沒有可保存的測試結果。請先到「🏠 基本回測」執行回測。")
            if st.button("➡️ 前往基本回測"):
                st.session_state['nav_to'] = "🏠 基本回測"
                st.rerun()
    
    with tab2:
        st.subheader("載入歷史測試")
        
        saved_tests = load_saved_tests()
        
        if saved_tests:
            st.info(f"📂 共有 {len(saved_tests)} 個已保存的測試")
            
            for i, test in enumerate(saved_tests):
                config = test.get('config', {})
                test_time = test.get('timestamp', 'Unknown')
                test_name = test.get('name', 'Unknown')
                
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.markdown(f"**{i+1}. {test_name}**")
                        st.caption(f"📍 {config.get('symbol', 'N/A')} | 🕐 {test_time[:16] if len(test_time) > 16 else test_time}")
                        strategies = config.get('strategies', [])
                        st.caption(f"📊 策略: {', '.join(strategies[:3])}{'...' if len(strategies) > 3 else ''}")
                    
                    with col2:
                        if st.button("📂 載入", key=f"load_{i}_{test_time}", use_container_width=True):
                            st.session_state['backtest_results'] = test.get('results', {})
                            st.session_state['symbol'] = config.get('symbol', 'Unknown')
                            st.success("✅ 已載入！前往「基本回測」查看")
                    
                    with col3:
                        if st.button("🗑️ 刪除", key=f"del_{i}_{test_time}", use_container_width=True):
                            try:
                                os.remove(test.get('filepath', ''))
                                st.success("已刪除")
                                st.rerun()
                            except Exception as e:
                                st.error(f"刪除失敗: {e}")
                    
                    st.divider()
            
            # 批量操作
            st.subheader("🔧 批量操作")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ 清空所有測試", type="secondary"):
                    st.warning("⚠️ 確定要刪除所有測試嗎？")
                    if st.button("確認刪除全部", key="confirm_delete_all"):
                        for test in saved_tests:
                            try:
                                os.remove(test.get('filepath', ''))
                            except:
                                pass
                        st.success("已清空所有測試")
                        st.rerun()
        else:
            st.info("📭 沒有已保存的測試。執行回測後可在此保存。")
    
    with tab3:
        st.subheader("快速匯出當前結果")
        
        if 'backtest_results' in st.session_state and st.session_state['backtest_results']:
            results = st.session_state['backtest_results']
            symbol = st.session_state.get('symbol', 'Unknown')
            
            # 準備匯出數據
            export_data = []
            for name, result in results.items():
                if hasattr(result, 'metrics'):
                    m = result.metrics
                    export_data.append({
                        '策略': name,
                        '市場': symbol,
                        '總投入': m.total_invested,
                        '最終價值': m.final_value,
                        '總報酬率(%)': round(m.total_return, 2),
                        'CAGR(%)': round(m.cagr, 2),
                        '夏普比率': round(m.sharpe_ratio, 3),
                        '最大回撤(%)': round(m.max_drawdown, 2),
                        '波動率(%)': round(m.volatility, 2),
                        '勝率(%)': round(m.win_rate, 2),
                        '投資月數': m.investment_months,
                    })
            
            if export_data:
                df = pd.DataFrame(export_data)
                st.dataframe(df, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # CSV 匯出
                    csv = df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="📥 下載 CSV",
                        data=csv,
                        file_name=f"DCA_{symbol}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # Excel 匯出
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name='績效總覽', index=False)
                        
                        # 加入交易記錄
                        for name, result in results.items():
                            if hasattr(result, 'to_transactions_df'):
                                tx_df = result.to_transactions_df()
                                if len(tx_df) > 0:
                                    # Excel sheet name 限制：移除非法字符並限制長度
                                    import re
                                    sheet_name = re.sub(r'[\[\]:*?/\\]', '_', name)[:31]
                                    tx_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    st.download_button(
                        label="📥 下載 Excel",
                        data=output.getvalue(),
                        file_name=f"DCA_{symbol}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
        else:
            st.info("💡 沒有可匯出的結果。請先執行回測。")


# ===================== 執行 =====================
if __name__ == "__main__":
    main()
