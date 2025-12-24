# pages/grid_search.py
"""
參數優化頁面
全自動網格搜索找出最佳配置
"""
import streamlit as st
import pandas as pd
from datetime import date, datetime
import plotly.express as px

from utils.common import get_data_loader
from utils.grid_search import GridSearchOptimizer, GridSearchConfig


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
