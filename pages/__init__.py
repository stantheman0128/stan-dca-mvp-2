# pages/__init__.py
"""
頁面模組
將各頁面功能拆分為獨立模組，提高可維護性
"""

from pages.basic_backtest import basic_backtest_page, display_results
from pages.robustness_test import robustness_test_page
from pages.cross_market import cross_market_page
from pages.grid_search import grid_search_page
from pages.test_management import test_management_page

__all__ = [
    'basic_backtest_page',
    'display_results',
    'robustness_test_page',
    'cross_market_page',
    'grid_search_page',
    'test_management_page',
]