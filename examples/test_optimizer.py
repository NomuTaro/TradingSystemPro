#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading System Pro - パラメータ最適化テスト
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from trading_system import TradingSystem
from extensions import Optimizer

def test_optimizer_basic():
    """基本的なOptimizer機能のテスト"""
    print("=== 基本テスト ===")
    
    optimizer = Optimizer(TradingSystem, stock_code="AAPL")
    
    # 小さなパラメータ範囲でテスト
    test_params = {
        'BUY_THRESHOLD': [2.0, 2.5],
        'SELL_THRESHOLD': [2.0]
    }
    
    print(f"パラメータ: {test_params}")
    results = optimizer.grid_search(param_ranges=test_params, objective="final_cash")
    
    if isinstance(results, pd.DataFrame) and len(results) > 0:
        best_result = results.nlargest(1, 'final_cash').iloc[0]
        print(f"✅ 成功! 最終資産: ${best_result['final_cash']:,.2f}")
        print(f"  収益率: {best_result['total_return']:.2%}")
        print(f"  取引回数: {best_result['trade_count']:.0f}")
        return True
    else:
        print("❌ 失敗")
        return False

def test_optimizer_advanced():
    """高度なOptimizer機能のテスト"""
    print("\n=== 高度なテスト ===")
    
    optimizer = Optimizer(TradingSystem, stock_code="AAPL")
    
    # 複数パラメータ最適化
    advanced_params = {
        'BUY_THRESHOLD': [1.5, 2.0],
        'SELL_THRESHOLD': [1.5, 2.0],
        'SIGNAL_WEIGHTS': {
            'golden_cross_short': [1.0, 1.5],
            'rsi_oversold': [1.0, 1.5]
        }
    }
    
    print(f"パラメータ: {advanced_params}")
    results = optimizer.grid_search(param_ranges=advanced_params, objective="final_cash")
    
    if isinstance(results, pd.DataFrame) and len(results) > 0:
        print(f"✅ 成功! 結果数: {len(results)}")
        
        # 統計情報
        print(f"最高資産: ${results['final_cash'].max():,.2f}")
        print(f"平均資産: ${results['final_cash'].mean():,.2f}")
        print(f"最高シャープ: {results['sharpe_ratio'].max():.3f}")
        
        # 最適結果
        best_cash = results.nlargest(1, 'final_cash').iloc[0]
        print(f"最適: ${best_cash['final_cash']:,.2f} (BUY={best_cash['BUY_THRESHOLD']}, SELL={best_cash['SELL_THRESHOLD']})")
        
        return True
    else:
        print("❌ 失敗")
        return False

def main():
    """メイン実行関数"""
    print("🚀 Trading System Pro - パラメータ最適化テスト")
    print("=" * 50)
    
    # テスト実行
    basic_success = test_optimizer_basic()
    advanced_success = test_optimizer_advanced()
    
    # 結果サマリー
    print("\n" + "=" * 50)
    print("🎯 結果サマリー")
    print("=" * 50)
    print(f"基本テスト: {'✅ 成功' if basic_success else '❌ 失敗'}")
    print(f"高度なテスト: {'✅ 成功' if advanced_success else '❌ 失敗'}")
    
    if basic_success and advanced_success:
        print("\n🎉 すべてのテストが成功しました！")
    else:
        print("\n❌ 一部のテストが失敗しました。")

if __name__ == "__main__":
    main()
