# core/visualizer.py
"""視覺化模組 - Plotly 互動圖表"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


# 金融事件標註
FINANCIAL_EVENTS = {
    '2008-09-15': '雷曼兄弟倒閉',
    '2008-10-10': '金融海嘯谷底',
    '2011-08-05': '美債降級',
    '2015-08-24': '中國股災',
    '2016-06-23': '英國脫歐',
    '2020-03-23': '疫情恐慌谷底',
    '2022-02-24': '俄烏戰爭',
    '2022-06-16': 'Fed 激進升息',
}


class Visualizer:
    """
    視覺化器
    
    提供 Plotly 互動式圖表生成功能
    """
    
    # 配色方案
    COLORS = [
        '#2196F3',  # 藍色
        '#4CAF50',  # 綠色
        '#FF9800',  # 橘色
        '#9C27B0',  # 紫色
        '#F44336',  # 紅色
        '#00BCD4',  # 青色
        '#795548',  # 棕色
        '#607D8B',  # 灰藍色
    ]
    
    def __init__(self, show_events: bool = True):
        """
        初始化視覺化器
        
        Args:
            show_events: 是否顯示金融事件標註
        """
        self.show_events = show_events
    
    def equity_curve_comparison(
        self,
        results: Dict[str, Any],
        title: str = "權益曲線對比",
        show_cost: bool = True
    ) -> go.Figure:
        """
        繪製多策略權益曲線對比圖
        
        Args:
            results: {策略名稱: BacktestResult} 字典
            title: 圖表標題
            show_cost: 是否顯示累積投入線
            
        Returns:
            Plotly Figure 物件
        """
        fig = go.Figure()
        
        for i, (name, result) in enumerate(results.items()):
            color = self.COLORS[i % len(self.COLORS)]
            df = result.equity_curve
            
            # 市值曲線
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['value'],
                name=name,
                line=dict(color=color, width=2),
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "日期: %{x}<br>"
                    "市值: %{y:,.0f}<br>"
                    "<extra></extra>"
                )
            ))
        
        # 累積投入線（以第一個策略為準）
        if show_cost and results:
            first_result = list(results.values())[0]
            df = first_result.equity_curve
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['cost'],
                name='累積投入',
                line=dict(color='gray', width=2, dash='dash'),
                hovertemplate=(
                    "<b>累積投入</b><br>"
                    "日期: %{x}<br>"
                    "金額: %{y:,.0f}<br>"
                    "<extra></extra>"
                )
            ))
        
        # 暫時停用金融事件標註（Plotly 版本兼容問題）
        # if self.show_events and results:
        #     self._add_event_annotations(fig, results)
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            xaxis_title="日期",
            yaxis_title="市值",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template='plotly_white',
            height=500
        )
        
        # 添加 range slider
        fig.update_xaxes(rangeslider_visible=True)
        
        return fig
    
    def return_curve(
        self,
        results: Dict[str, Any],
        title: str = "累積報酬率曲線"
    ) -> go.Figure:
        """
        繪製報酬率曲線（含正負區域填充）
        
        Args:
            results: {策略名稱: BacktestResult} 字典
            title: 圖表標題
            
        Returns:
            Plotly Figure 物件
        """
        fig = go.Figure()
        
        for i, (name, result) in enumerate(results.items()):
            color = self.COLORS[i % len(self.COLORS)]
            df = result.equity_curve
            
            # 報酬率曲線
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['return_pct'],
                name=name,
                line=dict(color=color, width=2),
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "日期: %{x}<br>"
                    "報酬率: %{y:.2f}%<br>"
                    "<extra></extra>"
                )
            ))
        
        # 零線
        fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            xaxis_title="日期",
            yaxis_title="報酬率 (%)",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def drawdown_curve(
        self,
        results: Dict[str, Any],
        title: str = "回撤曲線"
    ) -> go.Figure:
        """
        繪製回撤曲線
        
        Args:
            results: {策略名稱: BacktestResult} 字典
            title: 圖表標題
            
        Returns:
            Plotly Figure 物件
        """
        fig = go.Figure()
        
        for i, (name, result) in enumerate(results.items()):
            color = self.COLORS[i % len(self.COLORS)]
            df = result.equity_curve.copy()
            
            # 計算回撤
            values = df['value'].values
            peak = np.maximum.accumulate(values)
            drawdown = (values - peak) / peak * 100
            
            # 回撤曲線
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=drawdown,
                name=name,
                fill='tozeroy',
                line=dict(color=color, width=1),
                fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.3)',
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "日期: %{x}<br>"
                    "回撤: %{y:.2f}%<br>"
                    "<extra></extra>"
                )
            ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            xaxis_title="日期",
            yaxis_title="回撤 (%)",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def metrics_comparison_bar(
        self,
        results: Dict[str, Any],
        title: str = "績效指標對比"
    ) -> go.Figure:
        """
        繪製績效指標對比柱狀圖
        
        Args:
            results: {策略名稱: BacktestResult} 字典
            title: 圖表標題
            
        Returns:
            Plotly Figure 物件
        """
        # 準備數據
        metrics_to_show = [
            ('total_return', '總報酬率 (%)', True),  # (attr, label, higher_is_better)
            ('cagr', '年化報酬率 (%)', True),
            ('max_drawdown', '最大回撤 (%)', False),
            ('sharpe_ratio', '夏普比率', True),
            ('sortino_ratio', '索提諾比率', True),
            ('volatility', '年化波動率 (%)', False),
        ]
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[m[1] for m in metrics_to_show],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        strategy_names = list(results.keys())
        colors = [self.COLORS[i % len(self.COLORS)] for i in range(len(strategy_names))]
        
        for idx, (attr, label, higher_better) in enumerate(metrics_to_show):
            row = idx // 3 + 1
            col = idx % 3 + 1
            
            values = [getattr(results[name].metrics, attr) for name in strategy_names]
            
            # 找出最佳值的索引
            if higher_better:
                best_idx = np.argmax(values)
            else:
                best_idx = np.argmin([abs(v) for v in values])
            
            # 設定顏色（最佳值用金色標記）
            bar_colors = colors.copy()
            bar_colors[best_idx] = '#FFD700'  # 金色
            
            fig.add_trace(
                go.Bar(
                    x=strategy_names,
                    y=values,
                    marker_color=bar_colors,
                    showlegend=False,
                    hovertemplate=(
                        "%{x}<br>"
                        f"{label}: %{{y:.2f}}<br>"
                        "<extra></extra>"
                    )
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            template='plotly_white',
            height=500,
            showlegend=False
        )
        
        return fig
    
    def risk_return_scatter(
        self,
        results: Dict[str, Any],
        title: str = "風險-報酬分析"
    ) -> go.Figure:
        """
        繪製風險-報酬散點圖
        
        Args:
            results: {策略名稱: BacktestResult} 字典
            title: 圖表標題
            
        Returns:
            Plotly Figure 物件
        """
        fig = go.Figure()
        
        for i, (name, result) in enumerate(results.items()):
            color = self.COLORS[i % len(self.COLORS)]
            metrics = result.metrics
            
            # 點的大小根據夏普比率
            size = max(10, min(50, (metrics.sharpe_ratio + 1) * 15))
            
            fig.add_trace(go.Scatter(
                x=[metrics.volatility],
                y=[metrics.cagr],
                mode='markers+text',
                name=name,
                marker=dict(
                    size=size,
                    color=color,
                    line=dict(width=2, color='white')
                ),
                text=[name],
                textposition='top center',
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "年化報酬: %{y:.2f}%<br>"
                    "波動率: %{x:.2f}%<br>"
                    f"夏普比率: {metrics.sharpe_ratio:.2f}<br>"
                    "<extra></extra>"
                )
            ))
        
        # 添加理想區域標註（左上角）
        fig.add_annotation(
            x=0.05, y=0.95,
            xref="paper", yref="paper",
            text="↖ 理想區域<br>(低風險高報酬)",
            showarrow=False,
            font=dict(size=10, color="gray"),
            align="left"
        )
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            xaxis_title="年化波動率 (%) - 風險",
            yaxis_title="年化報酬率 (%) - 報酬",
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def monte_carlo_distribution(
        self,
        returns: List[float],
        title: str = "Monte Carlo 模擬結果分佈"
    ) -> go.Figure:
        """
        繪製 Monte Carlo 模擬結果分佈圖
        
        Args:
            returns: 報酬率列表
            title: 圖表標題
            
        Returns:
            Plotly Figure 物件
        """
        fig = go.Figure()
        
        # 直方圖
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=30,
            name='報酬分佈',
            marker_color='#2196F3',
            opacity=0.7,
            hovertemplate=(
                "報酬率區間: %{x}<br>"
                "次數: %{y}<br>"
                "<extra></extra>"
            )
        ))
        
        # 添加統計標線
        mean_ret = np.mean(returns)
        median_ret = np.median(returns)
        p5 = np.percentile(returns, 5)
        p95 = np.percentile(returns, 95)
        
        fig.add_vline(x=mean_ret, line_dash="solid", line_color="red", 
                     annotation_text=f"平均: {mean_ret:.1f}%")
        fig.add_vline(x=median_ret, line_dash="dash", line_color="green",
                     annotation_text=f"中位數: {median_ret:.1f}%")
        fig.add_vline(x=p5, line_dash="dot", line_color="orange",
                     annotation_text=f"5%: {p5:.1f}%")
        fig.add_vline(x=p95, line_dash="dot", line_color="orange",
                     annotation_text=f"95%: {p95:.1f}%")
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            xaxis_title="報酬率 (%)",
            yaxis_title="次數",
            template='plotly_white',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def _add_event_annotations(self, fig: go.Figure, results: Dict[str, Any]):
        """添加金融事件標註到圖表"""
        # 取得數據時間範圍
        first_result = list(results.values())[0]
        date_range = first_result.equity_curve['date']
        min_date = pd.to_datetime(date_range.min())
        max_date = pd.to_datetime(date_range.max())
        
        for event_date, event_name in FINANCIAL_EVENTS.items():
            event_dt = pd.to_datetime(event_date)
            if min_date <= event_dt <= max_date:
                # 轉換為字串格式避免 Plotly Timestamp 錯誤
                event_str = event_dt.strftime('%Y-%m-%d')
                fig.add_vline(
                    x=event_str,
                    line_dash="dot",
                    line_color="rgba(128, 128, 128, 0.5)",
                    annotation_text=event_name,
                    annotation_position="top",
                    annotation=dict(
                        font_size=8,
                        textangle=-90
                    )
                )
