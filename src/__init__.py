"""
TradingSystem Pro - 高度な株式取引分析・シミュレーションシステム

パッケージ初期化ファイル
"""

from .trading_system import TradingSystem
from . import config

__version__ = "1.0.0"
__author__ = "TradingSystem Pro Team"

# パッケージレベルでのインポート
__all__ = [
    'TradingSystem',
    'config'
]
