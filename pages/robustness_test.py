# pages/robustness_test.py
"""
穩健性測試頁面
測試策略在不同時間點和模擬情境下的表現穩定性
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime
from typing import Dict, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from strategies import (
    STRATEGY_REGISTRY,
    DCADipBuyingStrategy,
    DCATrendFilterStrategy,
)

from utils.common import (
    get_data_loader,
    get_backtest_engine,
    get_robustness_tester,
    create_strategy_instance,
)


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
        _fixed_start_test(data_loader, engine, symbol, strategy_name, base_amount, frequency)
    
    elif test_type == "🎲 Monte Carlo 模擬":
        _monte_carlo_test(data_loader, engine, symbol, strategy_name, base_amount, frequency)
    
    elif test_type == "📊 滾動窗口分析":
        _rolling_window_test(data_loader, engine, symbol, strategy_name, base_amount, frequency)
    
    elif test_type == "🎛️ 參數敏感度分析":
        _sensitivity_test(data_loader, engine, symbol, strategy_name, base_amount, frequency)


def _fixed_start_test(data_loader, engine, symbol, strategy_name, base_amount, frequency):
    """固定起始點測試"""
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
            fig = px.bar(df, x='起始日期', y='總報酬率(%)', 
                       title=f'{strategy_name} - 不同起始點總報酬率',
                       color='總報酬率(%)',
                       color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)


def _monte_carlo_test(data_loader, engine, symbol, strategy_name, base_amount, frequency):
    """Monte Carlo 隨機模擬"""
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


def _rolling_window_test(data_loader, engine, symbol, strategy_name, base_amount, frequency):
    """滾動窗口分析"""
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


def _sensitivity_test(data_loader, engine, symbol, strategy_name, base_amount, frequency):
    """參數敏感度分析"""
    st.subheader("參數敏感度分析")
    st.markdown("測試策略參數變化對結果的影響")
    
    # 只對 V1 跌深加碼策略進行參數敏感度分析
    if strategy_name == 'V1: 跌深加碼':
        _sensitivity_v1(data_loader, engine, symbol, base_amount, frequency)
    elif strategy_name == 'V2: 趨勢過濾':
        _sensitivity_v2(data_loader, engine, symbol, base_amount, frequency)
    else:
        st.info("請選擇 V1 跌深加碼 或 V2 趨勢過濾 策略進行參數敏感度分析")


def _sensitivity_v1(data_loader, engine, symbol, base_amount, frequency):
    """V1 策略敏感度分析"""
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


def _sensitivity_v2(data_loader, engine, symbol, base_amount, frequency):
    """V2 策略敏感度分析"""
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
            
            fig = px.line(df, x='參數值', y=['總報酬率(%)', '夏普比率'], 
                         title=f'V2 趨勢過濾 - {param_to_test} 敏感度分析')
            st.plotly_chart(fig, use_container_width=True)
