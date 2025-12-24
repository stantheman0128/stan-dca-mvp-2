# app.py
"""
定期定額策略回測工具 - Streamlit 主應用程式
DCA Strategy Backtesting Tool - Full Featured Version

重構版本 v2.3：頁面模組化，提高可維護性
"""
import streamlit as st
from pathlib import Path

# 設定頁面必須在最前面
st.set_page_config(
    page_title="DCA 策略回測工具",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 載入共用工具
from utils.common import (
    load_css,
    clear_memory_cache,
    get_memory_usage,
)

# 載入頁面模組
from pages import (
    basic_backtest_page,
    robustness_test_page,
    cross_market_page,
    grid_search_page,
    test_management_page,
)


def main():
    """主應用程式入口"""
    # 載入 CSS 樣式
    load_css()
    
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
    
    # 路由到對應頁面
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


# ===================== 執行 =====================
if __name__ == "__main__":
    main()
