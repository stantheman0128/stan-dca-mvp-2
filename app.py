# app.py
"""
定期定額策略回測工具 - Streamlit 主應用程式
DCA Strategy Backtesting Tool
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime
from typing import Dict, Any

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


# ===================== CSS 樣式 =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;600;700&display=swap');

html, body, [class*="css"], [class*="st-"] {
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


# ===================== 主應用 =====================
def main():
    # 標題
    st.title("📈 定期定額策略回測工具")
    st.caption("DCA Strategy Backtesting Tool - 研究不同定期定額策略的歷史表現")
    
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
    
    # 標籤頁
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 績效總覽", "📈 圖表分析", "📋 詳細數據", "🔬 統計檢驗", "📥 導出報告"
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
        st.subheader("📥 導出報告")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Excel 報告")
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
            st.markdown("### CSV 數據")
            st.markdown("下載對比表格數據")
            
            csv_data = report_gen.generate_comparison_table(results).to_csv()
            st.download_button(
                label="📥 下載 CSV 數據",
                data=csv_data,
                file_name=f"DCA_Comparison_{symbol}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )


# ===================== 執行 =====================
if __name__ == "__main__":
    main()
