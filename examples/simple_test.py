#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小限のパラメータ最適化例
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from trading_system import TradingSystem
from extensions import Optimizer

def simple_optimization():
    """最小限の最適化例"""
    print("🚀 パラメータ最適化開始")
    
    # 1. Optimizerを作成
    optimizer = Optimizer(TradingSystem, stock_code="AAPL")
    
    # 2. 最適化するパラメータを設定
    params = {
        'BUY_THRESHOLD': [2.0, 2.5],  # 買いシグナル閾値
        'SELL_THRESHOLD': [2.0]       # 売りシグナル閾値
    }
    
    print(f"パラメータ: {params}")
    print("最適化実行中...")
    
    # 3. 最適化を実行
    results = optimizer.grid_search(
        param_ranges=params,
        objective="final_cash"  # 最終資産を最大化
    )
    
    # 4. 結果を表示
    if len(results) > 0:
        best = results.nlargest(1, 'final_cash').iloc[0]
        print(f"\n✅ 最適化完了!")
        print(f"最良結果:")
        print(f"  最終資産: ${best['final_cash']:,.2f}")
        print(f"  収益率: {best['total_return']:.2%}")
        print(f"  最適パラメータ: BUY={best['BUY_THRESHOLD']}, SELL={best['SELL_THRESHOLD']}")
    else:
        print("❌ 結果が取得できませんでした")

if __name__ == "__main__":
    simple_optimization()
