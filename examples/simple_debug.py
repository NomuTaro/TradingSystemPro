#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from trading_system import TradingSystem

# 簡単なテスト実行
system = TradingSystem(stock_code="AAPL")
results = system.run_simulation()

print("=== 取引結果 ===")
print(f"初期資金: ${system.initial_cash:,.2f}")
print(f"最終資産: ${results['final_cash']:,.2f}")
print(f"収益率: {results['total_return']:.2%}")
print(f"シャープレシオ: {results['sharpe_ratio']:.3f}")

# 取引履歴の確認
if hasattr(system, 'trade_history'):
    print(f"取引回数: {len(system.trade_history)}")
    for i, trade in enumerate(system.trade_history[:3]):
        print(f"取引 {i+1}: {trade}")
