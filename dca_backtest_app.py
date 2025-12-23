# dca_backtest_app.py
# 定期定額回測計算器 MVP
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date

# 頁面設定
st.set_page_config(page_title="定期定額回測", page_icon="📈")

# 修正中文顯示（Streamlit UI + 表格/DataFrame + Matplotlib 圖表）
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;600&display=swap');

    html, body, [class*="css"], [class*="st-"] {
        font-family: "Noto Sans TC", "Microsoft JhengHei", "PingFang TC", "Heiti TC", "Noto Sans", sans-serif;
    }

    /* DataFrame / Data editor (前端資料表格) */
    div[data-testid="stDataFrame"],
    div[data-testid="stDataEditor"],
    .stDataFrame, .stDataEditor {
        font-family: "Noto Sans TC", "Microsoft JhengHei", "PingFang TC", "Heiti TC", "Noto Sans", sans-serif;
    }

    /* 自訂指標（避免大數字被截斷時太「無情」） */
    .metric-card { line-height: 1.15; }
    .metric-label { color: rgba(49, 51, 63, 0.7); font-size: 0.95rem; margin-bottom: 0.25rem; }
    .metric-value { font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    </style>
    """,
    unsafe_allow_html=True,
)

plt.rcParams["font.sans-serif"] = [
    "Noto Sans TC",
    "Microsoft JhengHei",
    "Microsoft YaHei",
    "PingFang TC",
    "Heiti TC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def _metric_font_size(value_text: str) -> str:
    text = value_text.strip()
    n = len(text)
    if n <= 12:
        return "2.25rem"
    if n <= 16:
        return "1.85rem"
    if n <= 20:
        return "1.55rem"
    return "1.35rem"


def render_metric(label: str, value: str) -> None:
    font_size = _metric_font_size(value)
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value" style="font-size: {font_size};">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.title("📈 定期定額回測計算器")
st.caption("模擬定期定額投資的歷史報酬表現")

# 股市選項
MARKETS = {
    "SPY (美股 S&P 500)": "SPY",
    "QQQ (美股 Nasdaq 100)": "QQQ",
    "0050.TW (台灣50)": "0050.TW",
    "^TWII (台灣加權指數)": "^TWII"
}

# 輸入區
col1, col2, col3 = st.columns(3)
with col1:
    market_name = st.selectbox("選擇市場", list(MARKETS.keys()))
with col2:
    start_date = st.date_input("開始日期", value=date(2015, 1, 1),
                                min_value=date(2010, 1, 1), max_value=date.today())
with col3:
    currency = "TWD" if "TW" in MARKETS[market_name] else "USD"
    monthly_invest = st.number_input(f"每月投入 ({currency})", 
                                      min_value=100, max_value=100000, value=1000, step=100)

def run_backtest(symbol: str, start: date, monthly: float) -> dict:
    """執行定期定額回測"""
    # 下載數據（auto_adjust=True 讓 Close 即為調整後價格）
    data = yf.download(symbol, start=start, end=date.today(), progress=False, auto_adjust=True)
    if data.empty:
        return None
    
    # 處理 MultiIndex columns（yfinance 0.2.40+ 的新格式）
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # 取每月第一個交易日（重採樣）
    monthly_data = data['Close'].resample('MS').first().dropna()
    if len(monthly_data) < 2:
        return None
    
    # 回測計算
    total_shares, total_cost = 0.0, 0.0
    history = []
    
    for dt, price in monthly_data.items():
        price_val = float(price)
        shares_bought = monthly / price_val
        total_shares += shares_bought
        total_cost += monthly
        current_value = total_shares * price_val
        return_pct = (current_value - total_cost) / total_cost * 100 if total_cost > 0 else 0
        
        history.append({
            'date': dt, 'price': price_val, 'total_cost': total_cost,
            'current_value': current_value, 'return_pct': return_pct
        })
    
    df = pd.DataFrame(history)
    months = len(df)
    years = months / 12
    final_return = (df['current_value'].iloc[-1] - df['total_cost'].iloc[-1]) / df['total_cost'].iloc[-1]
    annualized = ((1 + final_return) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    return {
        'df': df, 'months': months,
        'total_cost': df['total_cost'].iloc[-1],
        'final_value': df['current_value'].iloc[-1],
        'total_return': df['current_value'].iloc[-1] - df['total_cost'].iloc[-1],
        'return_pct': final_return * 100,
        'annualized': annualized
    }

# 執行回測
if st.button("🚀 開始回測", type="primary", use_container_width=True):
    symbol = MARKETS[market_name]
    
    with st.spinner("正在下載數據並計算..."):
        result = run_backtest(symbol, start_date, monthly_invest)
    
    if result is None:
        st.error("❌ 數據下載失敗或日期範圍無有效數據，請調整參數後重試。")
    else:
        currency = "TWD" if "TW" in symbol else "USD"
        
        # 關鍵指標
        st.subheader("📊 回測結果")
        c1, c2, c3 = st.columns(3)
        sign = "+" if result['total_return'] >= 0 else ""
        with c1:
            render_metric("總投入", f"{currency} {result['total_cost']:,.0f}")
        with c2:
            render_metric("最終市值", f"{currency} {result['final_value']:,.0f}")
        with c3:
            render_metric("總報酬", f"{sign}{currency} {result['total_return']:,.0f}")
        
        c4, c5, c6 = st.columns(3)
        with c4:
            render_metric("報酬率", f"{sign}{result['return_pct']:.2f}%")
        with c5:
            render_metric("年化報酬率", f"{result['annualized']:.2f}%")
        with c6:
            render_metric("投資期間", f"{result['months']} 個月")
        
        # 圖表
        df = result['df']
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 圖表1：累積投入 vs 市值
        ax1.plot(df['date'], df['total_cost'], label='累積投入', linestyle='--', color='gray')
        ax1.plot(df['date'], df['current_value'], label='投資市值', color='#2196F3')
        ax1.fill_between(df['date'], df['total_cost'], df['current_value'], 
                         where=df['current_value'] >= df['total_cost'], alpha=0.3, color='green')
        ax1.fill_between(df['date'], df['total_cost'], df['current_value'],
                         where=df['current_value'] < df['total_cost'], alpha=0.3, color='red')
        ax1.set_title('投資成長曲線', fontsize=14)
        ax1.set_ylabel(f'金額 ({currency})')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 圖表2：報酬率變化
        colors = ['green' if x >= 0 else 'red' for x in df['return_pct']]
        ax2.fill_between(df['date'], 0, df['return_pct'], 
                         where=df['return_pct'] >= 0, alpha=0.3, color='green')
        ax2.fill_between(df['date'], 0, df['return_pct'],
                         where=df['return_pct'] < 0, alpha=0.3, color='red')
        ax2.plot(df['date'], df['return_pct'], color='#333')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_title('累積報酬率變化', fontsize=14)
        ax2.set_ylabel('報酬率 (%)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.info(f"📅 回測期間：{df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {df['date'].iloc[-1].strftime('%Y-%m-%d')}")

# Footer
st.divider()
st.caption("⚠️ 此工具僅供參考，歷史績效不代表未來表現。數據來源：Yahoo Finance")