# pages/cross_market.py
"""
跨市場分析頁面
比較策略在不同市場的表現
"""
import streamlit as st
import pandas as pd
from datetime import date
import plotly.express as px

from strategies import STRATEGY_REGISTRY

from utils.common import (
    get_data_loader,
    get_backtest_engine,
    create_strategy_instance,
)


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
