# pages/test_management.py
"""
測試管理頁面
保存、載入、匯出和管理測試結果
"""
import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
import json
import io
import os
import re

from utils.common import (
    save_test_result,
    load_saved_tests,
)


def test_management_page():
    """測試管理頁面"""
    st.header("📁 測試管理")
    st.caption("保存、載入、匯出和管理測試結果")
    
    tab1, tab2, tab3 = st.tabs(["💾 保存測試", "📂 載入測試", "📥 快速匯出"])
    
    with tab1:
        _save_test_tab()
    
    with tab2:
        _load_test_tab()
    
    with tab3:
        _export_tab()


def _save_test_tab():
    """保存測試標籤頁"""
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


def _load_test_tab():
    """載入測試標籤頁"""
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


def _export_tab():
    """匯出標籤頁"""
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
