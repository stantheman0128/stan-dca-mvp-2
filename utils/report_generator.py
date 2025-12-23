# utils/report_generator.py
"""報告生成模組 - Excel 和 PDF 導出"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from io import BytesIO

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    from openpyxl.utils.dataframe import dataframe_to_rows
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False


class ReportGenerator:
    """
    報告生成器
    
    支援 Excel 多工作表導出
    """
    
    def __init__(self, output_dir: str = 'results/exports'):
        """
        初始化報告生成器
        
        Args:
            output_dir: 輸出目錄
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_excel_report(
        self,
        results: Dict[str, Any],
        filename: Optional[str] = None,
        return_bytes: bool = False
    ) -> Optional[BytesIO]:
        """
        生成 Excel 報告
        
        Args:
            results: {策略名稱: BacktestResult} 字典
            filename: 檔案名稱（不含副檔名）
            return_bytes: 是否返回 BytesIO（用於 Streamlit 下載）
            
        Returns:
            如果 return_bytes=True，返回 BytesIO；否則返回 None
        """
        if not OPENPYXL_AVAILABLE:
            raise ImportError("需要安裝 openpyxl: pip install openpyxl")
        
        wb = Workbook()
        
        # 樣式定義
        header_font = Font(bold=True, color='FFFFFF')
        header_fill = PatternFill(start_color='2196F3', end_color='2196F3', fill_type='solid')
        positive_fill = PatternFill(start_color='C8E6C9', end_color='C8E6C9', fill_type='solid')
        negative_fill = PatternFill(start_color='FFCDD2', end_color='FFCDD2', fill_type='solid')
        border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        # 工作表 1: 概述
        ws_summary = wb.active
        ws_summary.title = "概述"
        self._create_summary_sheet(ws_summary, results, header_font, header_fill, border)
        
        # 工作表 2: 詳細指標
        ws_metrics = wb.create_sheet("詳細指標")
        self._create_metrics_sheet(ws_metrics, results, header_font, header_fill, border, positive_fill, negative_fill)
        
        # 工作表 3-N: 每個策略的交易記錄
        for name, result in results.items():
            safe_name = name[:28].replace(':', '-')  # Excel 工作表名稱限制
            ws_trans = wb.create_sheet(f"交易-{safe_name}")
            self._create_transactions_sheet(ws_trans, result, header_font, header_fill, border)
        
        # 保存或返回
        if return_bytes:
            output = BytesIO()
            wb.save(output)
            output.seek(0)
            return output
        else:
            if filename is None:
                filename = f"DCA_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            filepath = self.output_dir / f"{filename}.xlsx"
            wb.save(filepath)
            return None
    
    def _create_summary_sheet(self, ws, results, header_font, header_fill, border):
        """建立概述工作表"""
        # 標題
        ws['A1'] = "定期定額策略回測報告"
        ws['A1'].font = Font(bold=True, size=16)
        ws.merge_cells('A1:E1')
        
        ws['A2'] = f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ws['A3'] = ""
        
        # 測試配置
        first_result = list(results.values())[0]
        ws['A4'] = "測試配置"
        ws['A4'].font = Font(bold=True)
        ws['A5'] = f"標的: {first_result.symbol}"
        ws['A6'] = f"期間: {first_result.start_date.strftime('%Y-%m-%d')} ~ {first_result.end_date.strftime('%Y-%m-%d')}"
        ws['A7'] = f"頻率: {'月' if first_result.frequency == 'M' else '週' if first_result.frequency == 'W' else '季'}"
        ws['A8'] = ""
        
        # 關鍵指標對比
        ws['A9'] = "關鍵指標對比"
        ws['A9'].font = Font(bold=True)
        
        headers = ['策略', '總報酬率(%)', '年化報酬率(%)', '最大回撤(%)', '夏普比率', '總投入', '最終市值']
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=10, column=col, value=header)
            cell.font = header_font
            cell.fill = header_fill
            cell.border = border
            cell.alignment = Alignment(horizontal='center')
        
        for row, (name, result) in enumerate(results.items(), 11):
            m = result.metrics
            values = [name, m.total_return, m.cagr, m.max_drawdown, m.sharpe_ratio, 
                     m.total_invested, m.final_value]
            for col, value in enumerate(values, 1):
                cell = ws.cell(row=row, column=col, value=value)
                cell.border = border
                if col == 1:
                    cell.alignment = Alignment(horizontal='left')
                else:
                    cell.alignment = Alignment(horizontal='right')
                    if isinstance(value, float):
                        cell.number_format = '#,##0.00'
        
        # 調整欄寬
        ws.column_dimensions['A'].width = 20
        for col in ['B', 'C', 'D', 'E', 'F', 'G']:
            ws.column_dimensions[col].width = 15
    
    def _create_metrics_sheet(self, ws, results, header_font, header_fill, border, pos_fill, neg_fill):
        """建立詳細指標工作表"""
        # 所有指標
        metric_labels = [
            ('total_return', '總報酬率 (%)'),
            ('total_return_amount', '總報酬金額'),
            ('cagr', '年化報酬率 (%)'),
            ('max_drawdown', '最大回撤 (%)'),
            ('max_drawdown_duration', '最長回撤期間 (期)'),
            ('volatility', '年化波動率 (%)'),
            ('downside_volatility', '下行波動率 (%)'),
            ('var_95', '95% VaR (%)'),
            ('var_99', '99% VaR (%)'),
            ('sharpe_ratio', '夏普比率'),
            ('sortino_ratio', '索提諾比率'),
            ('calmar_ratio', '卡爾馬比率'),
            ('total_invested', '總投入金額'),
            ('final_value', '最終市值'),
            ('total_periods', '總投入次數'),
            ('investment_months', '投資月數'),
            ('avg_cost', '平均持倉成本'),
            ('win_rate', '勝率 (%)'),
        ]
        
        # 表頭
        ws['A1'] = "指標"
        ws['A1'].font = header_font
        ws['A1'].fill = header_fill
        ws['A1'].border = border
        
        for col, name in enumerate(results.keys(), 2):
            cell = ws.cell(row=1, column=col, value=name)
            cell.font = header_font
            cell.fill = header_fill
            cell.border = border
            cell.alignment = Alignment(horizontal='center')
        
        # 指標數據
        for row, (attr, label) in enumerate(metric_labels, 2):
            ws.cell(row=row, column=1, value=label).border = border
            
            for col, (name, result) in enumerate(results.items(), 2):
                value = getattr(result.metrics, attr)
                cell = ws.cell(row=row, column=col, value=value)
                cell.border = border
                cell.alignment = Alignment(horizontal='right')
                
                if isinstance(value, float):
                    cell.number_format = '#,##0.00'
        
        # 調整欄寬
        ws.column_dimensions['A'].width = 25
        for i, _ in enumerate(results.keys(), 2):
            ws.column_dimensions[chr(64 + i)].width = 18
    
    def _create_transactions_sheet(self, ws, result, header_font, header_fill, border):
        """建立交易記錄工作表"""
        df = result.to_transactions_df()
        
        if df.empty:
            ws['A1'] = "無交易記錄"
            return
        
        # 表頭
        for col, header in enumerate(df.columns, 1):
            cell = ws.cell(row=1, column=col, value=header)
            cell.font = header_font
            cell.fill = header_fill
            cell.border = border
            cell.alignment = Alignment(horizontal='center')
        
        # 數據
        for row_idx, row in enumerate(df.itertuples(index=False), 2):
            for col_idx, value in enumerate(row, 1):
                cell = ws.cell(row=row_idx, column=col_idx, value=value)
                cell.border = border
                
                # 日期格式
                if col_idx == 1 and hasattr(value, 'strftime'):
                    cell.number_format = 'YYYY-MM-DD'
                # 數字格式
                elif isinstance(value, float):
                    cell.number_format = '#,##0.00'
        
        # 調整欄寬
        for i, col in enumerate(df.columns, 1):
            ws.column_dimensions[chr(64 + i)].width = max(12, len(str(col)) + 2)
    
    def generate_comparison_table(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        生成策略對比表格
        
        Args:
            results: {策略名稱: BacktestResult} 字典
            
        Returns:
            對比表格 DataFrame
        """
        rows = []
        for name, result in results.items():
            m = result.metrics
            rows.append({
                '策略': name,
                '總報酬率(%)': m.total_return,
                '年化報酬率(%)': m.cagr,
                '最大回撤(%)': m.max_drawdown,
                '夏普比率': m.sharpe_ratio,
                '索提諾比率': m.sortino_ratio,
                '卡爾馬比率': m.calmar_ratio,
                '年化波動率(%)': m.volatility,
                '勝率(%)': m.win_rate,
                '總投入': m.total_invested,
                '最終市值': m.final_value,
            })
        
        df = pd.DataFrame(rows)
        return df.set_index('策略')
