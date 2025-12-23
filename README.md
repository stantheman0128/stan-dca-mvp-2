# 定期定額策略回測工具
# DCA Strategy Backtesting Tool

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

## 📝 專案概述

這是一個專業的定期定額（DCA, Dollar-Cost Averaging）策略回測工具，支援多種策略變化、跨市場回測、完整績效指標和報告導出。

## ✨ 功能特點

- **多策略對比**: V0~V3 四種策略變化同時比較
- **20年回測**: 支援 2005-2025 年歷史數據
- **跨市場**: 美股 (SPY, QQQ, VTI)、台股 (0050.TW, 0056.TW)、國際市場
- **完整指標**: 總報酬率、年化報酬率、夏普比率、最大回撤等 18 項指標
- **互動圖表**: Plotly 互動式視覺化（權益曲線、回撤、對比）
- **統計檢驗**: t 檢驗、信賴區間、多策略比較
- **報告導出**: Excel 多工作表報告下載

## 📊 支援策略

| 策略 | 說明 | 關鍵參數 |
|------|------|----------|
| **V0: 純定期定額** | 固定金額定期投入，作為基準 | base_amount |
| **V1: 跌深加碼** | 價格相對高點下跌時增加投入 | dip_threshold, multiplier |
| **V2: 趨勢過濾** | 價格低於移動平均線時加碼 | ma_period, ma_type |
| **V3: 波動率調整** | 高波動環境時加碼買入 | volatility_window, threshold |

## 🚀 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 執行應用

```bash
streamlit run app.py
```

### 3. 開始使用

1. 在左側選擇市場和時間範圍
2. 選擇要比較的策略
3. 調整策略參數（可選）
4. 點擊「開始回測」
5. 查看結果和圖表
6. 下載報告

## 📁 專案結構

```
dca-2/
├── app.py                 # Streamlit 主應用
├── requirements.txt       # 依賴項
├── README.md             # 專案說明
├── core/                  # 核心模組
│   ├── data_loader.py    # 數據下載與快取
│   ├── backtest_engine.py # 回測引擎
│   ├── metrics.py        # 績效指標計算
│   ├── statistics.py     # 統計檢驗
│   └── visualizer.py     # 圖表視覺化
├── strategies/            # 策略模組
│   ├── base_strategy.py  # 策略基類
│   ├── dca_pure.py       # V0: 純 DCA
│   ├── dca_dip_buying.py # V1: 跌深加碼
│   ├── dca_trend_filter.py # V2: 趨勢過濾
│   └── dca_volatility.py # V3: 波動率調整
├── utils/                 # 工具模組
│   ├── report_generator.py # 報告生成
│   └── robustness_tester.py # 穩健性測試
├── data/                  # 快取數據
└── results/              # 輸出報告
```

## 📈 績效指標

### 報酬指標
- 總報酬率、年化報酬率 (CAGR)
- 最終市值、總投入金額

### 風險指標
- 年化波動率、最大回撤
- 回撤天數、波動率

### 風險調整後報酬
- 夏普比率 (Sharpe Ratio)
- 索提諾比率 (Sortino Ratio)
- 卡爾馬比率 (Calmar Ratio)

### 其他指標
- 勝率、盈虧比
- 平均成本、買入次數

## 🔧 自訂策略

繼承 `BaseStrategy` 並實作 `decide` 方法：

```python
from strategies.base_strategy import BaseStrategy, InvestmentDecision

class MyCustomStrategy(BaseStrategy):
    name = "我的策略"
    version = "1.0"
    
    def decide(self, date, price, historical_data, context):
        # 你的策略邏輯
        return InvestmentDecision(
            should_invest=True,
            amount=self.base_amount,
            multiplier=1.0,
            reason="正常買入"
        )
```

## 📊 統計檢驗

- **配對 t 檢驗**: 比較兩策略報酬差異顯著性
- **信賴區間**: 95% CI 估計平均超額報酬
- **多重比較**: Bonferroni 校正後的 p 值

## 🤝 授權

MIT License

## 📞 聯絡方式

GitHub: [@stantheman0128](https://github.com/stantheman0128)
