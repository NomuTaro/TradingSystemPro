#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from trading_system import TradingSystem

# 簡単なテスト実行
print("=== 修正後の収益率確認 ===")
system = TradingSystem(stock_code="AAPL")
results = system.run_simulation()

print(f"初期資金: ${system.initial_cash:,.2f}")
print(f"最終資産: ${results['final_cash']:,.2f}")
print(f"収益率: {results['total_return']:.2%}")
print(f"シャープレシオ: {results['sharpe_ratio']:.3f}")
print(f"最大ドローダウン: {results['max_drawdown']:.2%}")

# 取引数の確認
if hasattr(system, 'trade_history'):
    buy_trades = [t for t in system.trade_history if t['type'] == 'BUY']
    sell_trades = [t for t in system.trade_history if t['type'] == 'SELL']
    print(f"取引回数: 買い {len(buy_trades)}, 売り {len(sell_trades)}")
    
    # 最初の数件の取引を表示
    if buy_trades:
        print(f"最初の買い取引: {buy_trades[0]['date']} ${buy_trades[0]['price']:.2f} x {buy_trades[0]['qty']}")
    if sell_trades:
        print(f"最初の売り取引: {sell_trades[0]['date']} ${sell_trades[0]['price']:.2f} x {sell_trades[0]['qty']}")
