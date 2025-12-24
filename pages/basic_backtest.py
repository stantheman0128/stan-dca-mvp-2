# pages/basic_backtest.py
"""
基本回測頁面
DCA 策略基本回測功能
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime
from typing import Dict, Any
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

from core.data_loader import DataLoader
from core.backtest_engine import BacktestEngine
from core.statistics import StatisticsCalculator
from strategies import STRATEGY_REGISTRY

from utils.common import (
    get_data_loader,
    get_backtest_engine,
    get_visualizer,
    get_report_generator,
    create_strategy_instance,
    render_metric_card,
    calculate_buy_and_hold,
)


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
                    '回撤減少(%)': bh_metrics['max_drawdown'] - m.max_drawdown,
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
