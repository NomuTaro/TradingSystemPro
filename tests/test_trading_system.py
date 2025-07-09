"""
TradingSystem Pro テストファイル

基本的なユニットテストを提供します。
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# パッケージパスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import TradingSystem, config
from src.trading_system import (
    detect_double_top, 
    detect_double_bottom,
    detect_three_gap_up,
    detect_three_gap_down,
    detect_three_white_soldiers,
    detect_three_black_crows
)

class TestTradingSystem:
    """TradingSystemクラスのテスト"""
    
    def setup_method(self):
        """各テストメソッド前の準備"""
        self.trading_system = TradingSystem()
    
    def test_init(self):
        """初期化テスト"""
        assert self.trading_system.stock_code == config.STOCK_CODE
        assert self.trading_system.initial_cash == config.INITIAL_CASH
        assert self.trading_system.df is None
        assert self.trading_system.asset_history == []
        assert self.trading_system.trade_history == []
    
    def test_standardize_column_names(self):
        """カラム名標準化テスト"""
        # テスト用データフレーム作成
        test_df = pd.DataFrame({
            'Close': [100, 101, 102],
            'SMA_5': [99, 100, 101],
            'RSI_14': [50, 55, 60],
            'MACD_12_26_9': [0.1, 0.2, 0.3],
            'BBU_25_2.0': [105, 106, 107]
        })
        
        self.trading_system._standardize_column_names(test_df)
        
        assert 'sma5' in test_df.columns
        assert 'RSI' in test_df.columns
        assert 'MACD' in test_df.columns
        assert 'BB_upper' in test_df.columns
    
    def test_calculate_win_rate(self):
        """勝率計算テスト"""
        # テスト用取引履歴
        test_trades = [
            {'type': 'BUY', 'date': pd.Timestamp('2023-01-01'), 'price': 100, 'qty': 10},
            {'type': 'SELL', 'date': pd.Timestamp('2023-01-02'), 'price': 110, 'qty': 10},
            {'type': 'BUY', 'date': pd.Timestamp('2023-01-03'), 'price': 110, 'qty': 10},
            {'type': 'SELL', 'date': pd.Timestamp('2023-01-04'), 'price': 105, 'qty': 10}
        ]
        
        win_rate = self.trading_system._calculate_win_rate(test_trades)
        assert win_rate == 50.0  # 1勝1敗なので50%
    
    def test_calculate_profit_factor(self):
        """プロフィットファクター計算テスト"""
        # テスト用取引履歴
        test_trades = [
            {'type': 'BUY', 'date': pd.Timestamp('2023-01-01'), 'price': 100, 'qty': 10},
            {'type': 'SELL', 'date': pd.Timestamp('2023-01-02'), 'price': 110, 'qty': 10},
            {'type': 'BUY', 'date': pd.Timestamp('2023-01-03'), 'price': 110, 'qty': 10},
            {'type': 'SELL', 'date': pd.Timestamp('2023-01-04'), 'price': 90, 'qty': 10}
        ]
        
        profit_factor = self.trading_system._calculate_profit_factor(test_trades)
        assert profit_factor > 0

class TestPatternDetection:
    """酒田五法パターン検出関数のテスト"""
    
    def test_detect_double_top(self):
        """ダブルトップ検出テスト"""
        # ダブルトップパターンのテストデータ
        prices = pd.Series([100, 110, 105, 109, 100], 
                          index=pd.date_range('2023-01-01', periods=5))
        
        # パターンが検出されるかテスト
        result = detect_double_top(prices, threshold=0.05)
        assert isinstance(result, bool)
    
    def test_detect_double_bottom(self):
        """ダブルボトム検出テスト"""
        # ダブルボトムパターンのテストデータ
        prices = pd.Series([100, 90, 95, 91, 100], 
                          index=pd.date_range('2023-01-01', periods=5))
        
        result = detect_double_bottom(prices, threshold=0.05)
        assert isinstance(result, bool)
    
    def test_detect_three_gap_up(self):
        """三空踏み上げ検出テスト"""
        # テスト用データ（窓開け上昇パターン）
        test_df = pd.DataFrame({
            'High': [100, 105, 110, 115, 120],
            'Low': [95, 102, 107, 112, 117]
        })
        
        result = detect_three_gap_up(test_df)
        assert isinstance(result, bool)
    
    def test_detect_three_white_soldiers(self):
        """赤三兵検出テスト"""
        # 3日連続陽線のテストデータ
        test_df = pd.DataFrame({
            'Open': [100, 102, 104],
            'Close': [102, 104, 106]
        })
        
        result = detect_three_white_soldiers(test_df)
        assert isinstance(result, bool)

class TestConfigIntegration:
    """設定ファイル統合テスト"""
    
    def test_config_values(self):
        """設定値の存在確認"""
        assert hasattr(config, 'STOCK_CODE')
        assert hasattr(config, 'INITIAL_CASH')
        assert hasattr(config, 'BUY_THRESHOLD')
        assert hasattr(config, 'SELL_THRESHOLD')
        assert hasattr(config, 'SIGNAL_WEIGHTS')
        
        assert isinstance(config.INITIAL_CASH, int)
        assert isinstance(config.BUY_THRESHOLD, (int, float))
        assert isinstance(config.SIGNAL_WEIGHTS, dict)
    
    def test_signal_weights_completeness(self):
        """シグナル重みの完全性確認"""
        required_signals = [
            'golden_cross_short', 'golden_cross_long',
            'bb_oversold', 'rsi_oversold', 'macd_bullish',
            'double_bottom', 'three_white_soldiers', 'three_gap_down',
            'dead_cross_short', 'dead_cross_long',
            'bb_overbought', 'rsi_overbought', 'macd_bearish',
            'double_top', 'three_black_crows', 'three_gap_up'
        ]
        
        for signal in required_signals:
            assert signal in config.SIGNAL_WEIGHTS

class TestDataHandling:
    """データ処理テスト"""
    
    def test_performance_metrics_calculation(self):
        """パフォーマンス指標計算テスト"""
        # テスト用資産履歴
        test_asset_history = [
            (pd.Timestamp('2023-01-01'), 1000000),
            (pd.Timestamp('2023-01-02'), 1050000),
            (pd.Timestamp('2023-01-03'), 980000),
            (pd.Timestamp('2023-01-04'), 1020000)
        ]
        
        trading_system = TradingSystem()
        metrics = trading_system._calculate_performance_metrics(test_asset_history)
        
        assert 'max_drawdown' in metrics
        assert 'sharpe_ratio' in metrics
        assert isinstance(metrics['max_drawdown'], (int, float))
        assert isinstance(metrics['sharpe_ratio'], (int, float))

# フィクスチャ
@pytest.fixture
def sample_price_data():
    """サンプル価格データを提供"""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)  # 再現性のため
    
    prices = []
    price = 100
    for _ in range(100):
        change = np.random.normal(0, 0.02)  # 2%の標準偏差
        price *= (1 + change)
        prices.append(price)
    
    return pd.DataFrame({
        'Date': dates,
        'Open': np.array(prices) * 0.99,
        'High': np.array(prices) * 1.01,
        'Low': np.array(prices) * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, 100)
    }).set_index('Date')

# 統合テスト
def test_full_system_integration(sample_price_data):
    """システム全体の統合テスト"""
    trading_system = TradingSystem()
    
    # サンプルデータを直接設定
    trading_system.df = sample_price_data
    
    # 基本的なテクニカル指標を追加
    trading_system.df['sma5'] = trading_system.df['Close'].rolling(5).mean()
    trading_system.df['sma25'] = trading_system.df['Close'].rolling(25).mean()
    trading_system.df['sma75'] = trading_system.df['Close'].rolling(75).mean()
    trading_system.df['RSI'] = 50  # 簡略化
    trading_system.df['MACD'] = 0
    trading_system.df['MACD_signal'] = 0
    trading_system.df['BB_upper'] = trading_system.df['Close'] * 1.02
    trading_system.df['BB_lower'] = trading_system.df['Close'] * 0.98
    trading_system.df['ATR'] = trading_system.df['Close'] * 0.01
    
    # シミュレーション実行
    results = trading_system.run_simulation()
    
    # 結果検証
    assert isinstance(results, dict)
    assert 'final_cash' in results
    assert 'total_return_pct' in results
    assert 'asset_history' in results
    assert 'trade_history' in results

if __name__ == "__main__":
    # テスト実行
    pytest.main([__file__, "-v"])
