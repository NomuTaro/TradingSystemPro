#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading System Pro - 取引デバッグスクリプト

取引の詳細を確認して負の収益率の原因を調査します。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from trading_system import TradingSystem

def debug_trading_system():
    """取引システムの詳細デバッグ"""
    print("=== Trading System Pro - 取引デバッグ ===\n")
    
    try:
        # 1. 基本設定でシステム作成
        print("1. 取引システム作成...")
        system = TradingSystem(stock_code="AAPL")
        
        # 2. 簡単なシミュレーション実行
        print("2. シミュレーション実行...")
        results = system.run_simulation()
        
        # 3. 取引履歴の詳細確認
        print("3. 取引履歴の詳細確認...")
        if hasattr(system, 'trade_history') and system.trade_history:
            print(f"取引回数: {len(system.trade_history)}")
            
            # 最初の5件の取引を表示
            print("\n最初の5件の取引:")
            for i, trade in enumerate(system.trade_history[:5]):
                print(f"  取引 {i+1}: {trade}")
            
            # 取引統計
            buy_trades = [t for t in system.trade_history if t['action'] == 'BUY']
            sell_trades = [t for t in system.trade_history if t['action'] == 'SELL']
            print(f"\n取引統計:")
            print(f"  買い注文: {len(buy_trades)}")
            print(f"  売り注文: {len(sell_trades)}")
            
            # 収益計算
            if buy_trades and sell_trades:
                profits = []
                for sell in sell_trades:
                    # 対応する買い注文を探す
                    sell_date = pd.to_datetime(sell['date'])
                    matching_buys = [b for b in buy_trades if pd.to_datetime(b['date']) < sell_date]
                    if matching_buys:
                        last_buy = matching_buys[-1]
                        profit = (sell['price'] - last_buy['price']) * sell['quantity']
                        profits.append(profit)
                        print(f"    取引ペア: 買い${last_buy['price']:.2f} -> 売り${sell['price']:.2f}, 損益: ${profit:.2f}")
                
                if profits:
                    total_profit = sum(profits)
                    print(f"\n合計損益: ${total_profit:.2f}")
                    print(f"平均損益: ${np.mean(profits):.2f}")
        
        # 4. 最終結果
        print(f"\n4. 最終結果:")
        print(f"  初期資金: ${system.initial_cash:,.2f}")
        print(f"  最終資産: ${results['final_cash']:,.2f}")
        print(f"  収益率: {results['total_return']:.2%}")
        print(f"  シャープレシオ: {results['sharpe_ratio']:.3f}")
        print(f"  最大ドローダウン: {results['max_drawdown']:.2%}")
        
        # 5. 日次資産価値の確認
        if hasattr(system, 'portfolio_values') and system.portfolio_values:
            print(f"\n5. 日次資産価値サンプル:")
            values = list(system.portfolio_values.values())
            dates = list(system.portfolio_values.keys())
            
            print(f"  開始: {dates[0]} -> ${values[0]:,.2f}")
            if len(values) > 1:
                print(f"  中間: {dates[len(dates)//2]} -> ${values[len(values)//2]:,.2f}")
                print(f"  終了: {dates[-1]} -> ${values[-1]:,.2f}")
            
            # 最低値と最高値
            min_value = min(values)
            max_value = max(values)
            print(f"  最低資産価値: ${min_value:,.2f}")
            print(f"  最高資産価値: ${max_value:,.2f}")
        
        # 6. 設定パラメータの確認
        print(f"\n6. 現在の設定パラメータ:")
        config = system.config
        print(f"  BUY_THRESHOLD: {config.BUY_THRESHOLD}")
        print(f"  SELL_THRESHOLD: {config.SELL_THRESHOLD}")
        print(f"  INITIAL_CASH: ${config.INITIAL_CASH:,.2f}")
        print(f"  TRANSACTION_COST_RATE: {config.TRANSACTION_COST_RATE:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ デバッグエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_trading_system()
