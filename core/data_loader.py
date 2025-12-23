# core/data_loader.py
"""數據下載與緩存模組"""
import os
import pickle
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Union

import pandas as pd
import yfinance as yf


class DataLoader:
    """
    數據載入器：負責下載、緩存和管理市場數據
    
    支援市場:
    - 美股: SPY, QQQ, DIA, IWM
    - 台股: 0050.TW, 0056.TW, ^TWII
    - 國際: ^N225, ^FTSE, ^GDAXI
    """
    
    # 預設支援的市場
    SUPPORTED_MARKETS = {
        # 美股
        'SPY': 'S&P 500 ETF',
        'QQQ': 'Nasdaq 100 ETF',
        'DIA': 'Dow Jones ETF',
        'IWM': 'Russell 2000 ETF',
        # 台股
        '0050.TW': '元大台灣50',
        '0056.TW': '元大高股息',
        '^TWII': '台灣加權指數',
        # 國際
        '^N225': '日經225',
        '^FTSE': '英國富時100',
        '^GDAXI': '德國DAX',
    }
    
    def __init__(self, cache_dir: str = 'data/cache'):
        """
        初始化數據載入器
        
        Args:
            cache_dir: 緩存目錄路徑
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_path(self, symbol: str) -> Path:
        """取得緩存檔案路徑"""
        safe_symbol = symbol.replace('^', '_').replace('.', '_')
        return self.cache_dir / f"{safe_symbol}_data.pkl"
    
    def _is_cache_fresh(self, cache_path: Path, max_age_days: int = 1) -> bool:
        """檢查緩存是否新鮮"""
        if not cache_path.exists():
            return False
        
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - mtime
        return age.days < max_age_days
    
    def download(
        self,
        symbol: str,
        start_date: Union[str, date] = '2005-01-01',
        end_date: Union[str, date, None] = None,
        use_cache: bool = True,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        下載或從緩存讀取市場數據
        
        Args:
            symbol: 股票代碼（如 SPY, 0050.TW）
            start_date: 開始日期
            end_date: 結束日期（預設今天）
            use_cache: 是否使用緩存
            force_refresh: 強制重新下載
            
        Returns:
            DataFrame，包含 OHLCV 數據
            
        Raises:
            ValueError: 數據下載失敗或無數據
        """
        if end_date is None:
            end_date = date.today()
            
        # 轉換日期格式
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
            
        cache_path = self._get_cache_path(symbol)
        
        # 嘗試使用緩存
        if use_cache and not force_refresh and self._is_cache_fresh(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    
                # 檢查緩存數據是否涵蓋請求範圍
                if (cached_data.index.min().date() <= start_date and 
                    cached_data.index.max().date() >= end_date - timedelta(days=5)):
                    # 篩選請求範圍
                    mask = (cached_data.index.date >= start_date) & (cached_data.index.date <= end_date)
                    return cached_data[mask].copy()
            except Exception:
                pass  # 緩存讀取失敗，重新下載
        
        # 下載數據
        data = self._download_with_retry(symbol, start_date, end_date)
        
        # 保存緩存
        if use_cache and not data.empty:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
            except Exception:
                pass  # 緩存保存失敗不影響主流程
                
        return data
    
    def _download_with_retry(
        self, 
        symbol: str, 
        start_date: date, 
        end_date: date,
        max_retries: int = 3
    ) -> pd.DataFrame:
        """帶重試機制的數據下載"""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                data = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date + timedelta(days=1),  # yfinance end 是 exclusive
                    progress=False,
                    auto_adjust=True
                )
                
                if data.empty:
                    raise ValueError(f"No data available for {symbol}")
                
                # 處理 MultiIndex columns (yfinance 0.2.40+)
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                # 確保索引是 DatetimeIndex
                data.index = pd.to_datetime(data.index)
                
                # 前向填充缺失值
                data = data.ffill()
                
                return data
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    continue
                    
        raise ValueError(
            f"數據下載失敗: {symbol}\n"
            f"錯誤: {str(last_error)}\n\n"
            f"可能原因:\n"
            f"• 網路連線問題\n"
            f"• 標的代碼錯誤（請確認格式，如: SPY, 0050.TW）\n"
            f"• 日期範圍無數據"
        )
    
    def get_available_markets(self) -> Dict[str, str]:
        """取得支援的市場清單"""
        return self.SUPPORTED_MARKETS.copy()
    
    def clear_cache(self, symbol: Optional[str] = None):
        """清除緩存"""
        if symbol:
            cache_path = self._get_cache_path(symbol)
            if cache_path.exists():
                cache_path.unlink()
        else:
            for f in self.cache_dir.glob('*.pkl'):
                f.unlink()
