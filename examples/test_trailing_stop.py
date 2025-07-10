#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading System Pro - トレーリングストップ機能テスト

このスクリプトは、新しく追加されたATRベースのトレーリングストップ機能をテストします。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from trading_system import TradingSystem
import config
import warnings

warnings.simplefilter('ignore')

def test_trailing_stop():
    """トレーリングストップ機能のテスト"""
    print("=== Trading System Pro - トレーリングストップ機能テスト ===\n")
    
    # 設定
    stock_code = "7203.JP"  # トヨタ自動車
    initial_cash = 1_000_000  # 100万円
    
    print(f"銘柄: {stock_code}")
    print(f"初期資金: {initial_cash:,.0f}円")
    print("="*50)
    
    # TradingSystemインスタンスを作成
    system = TradingSystem(stock_code=stock_code)
    system.initial_cash = initial_cash
    
    # データを準備
    print("データ準備中...")
    df = system.prepare_data()
    if df is None:
        print("データ取得に失敗しました。")
        return
    
    print(f"データ取得完了: {len(df)}件")
    # 期間表示の修正
    import pandas as pd
    from pandas._libs.tslibs.nattype import NaTType
    def safe_date_str(x):
        if isinstance(x, (pd.Series, pd.Index, pd.DataFrame)):
            return ''
        from pandas._libs.tslibs.nattype import NaTType
        if isinstance(x, NaTType):
            return ''
        if pd.isna(x):
            return ''
        try:
            ts = pd.Timestamp(x)
            if isinstance(ts, NaTType) or pd.isna(ts):
                return ''
            return ts.strftime('%Y-%m-%d')
        except Exception:
            return str(x)[:10]
    idx_list = df.index.to_list()
    start_str = safe_date_str(idx_list[0]) if idx_list else ''
    end_str = safe_date_str(idx_list[-1]) if idx_list else ''
    print(f"期間: {start_str} ~ {end_str}")
    
    # シミュレーション実行
    print("\nトレーリングストップ機能付きシミュレーション実行中...")
    asset_history, trade_history, final_cash = system.run_simulation()
    
    if asset_history is None or trade_history is None:
        print("シミュレーションに失敗しました。")
        return
    
    # トレーリングストップの分析
    print("\n📊 トレーリングストップ分析")
    print("="*50)
    
    trailing_stop_trades = []
    other_trades = []
    
    for i, trade in enumerate(trade_history):
        if trade['type'] == 'BUY':
            # 次の売り取引を探す
            if i + 1 < len(trade_history) and trade_history[i + 1]['type'] == 'SELL':
                sell_trade = trade_history[i + 1]
                
                # トレーリングストップによる売りかどうかチェック
                if 'Trailing Stop' in sell_trade.get('reason', ''):
                    trailing_stop_trades.append({
                        'buy_date': trade['date'],
                        'buy_price': trade['price'],
                        'sell_date': sell_trade['date'],
                        'sell_price': sell_trade['price'],
                        'initial_stop': trade.get('initial_stop', 0),
                        'final_stop': sell_trade.get('final_stop', 0),
                        'profit': sell_trade['proceeds'] - trade['cost']
                    })
                else:
                    other_trades.append({
                        'buy_date': trade['date'],
                        'buy_price': trade['price'],
                        'sell_date': sell_trade['date'],
                        'sell_price': sell_trade['price'],
                        'reason': sell_trade.get('reason', 'Unknown'),
                        'profit': sell_trade['proceeds'] - trade['cost']
                    })
    
    # 結果表示
    print(f"総取引数: {len(trade_history) // 2}回")
    print(f"トレーリングストップによる売り: {len(trailing_stop_trades)}回")
    print(f"その他の売り: {len(other_trades)}回")
    
    if trailing_stop_trades:
        print(f"\n🎯 トレーリングストップ取引詳細:")
        total_trailing_profit = 0
        for i, trade in enumerate(trailing_stop_trades, 1):
            profit = trade['profit']
            total_trailing_profit += profit
            stop_improvement = trade['final_stop'] - trade['initial_stop']
            
            print(f"\n{i}回目:")
            print(f"  買い: {trade['buy_date'].strftime('%y-%m-%d')} {trade['buy_price']:,.0f}円")
            print(f"  売り: {trade['sell_date'].strftime('%y-%m-%d')} {trade['sell_price']:,.0f}円")
            print(f"  初期ストップ: {trade['initial_stop']:,.0f}円")
            print(f"  最終ストップ: {trade['final_stop']:,.0f}円")
            print(f"  ストップ改善: {stop_improvement:,.0f}円")
            print(f"  損益: {profit:,.0f}円")
        
        print(f"\nトレーリングストップ取引の総損益: {total_trailing_profit:,.0f}円")
    
    if other_trades:
        print(f"\n📈 その他の取引:")
        total_other_profit = 0
        for i, trade in enumerate(other_trades, 1):
            profit = trade['profit']
            total_other_profit += profit
            
            print(f"\n{i}回目 ({trade['reason']}):")
            print(f"  買い: {trade['buy_date'].strftime('%y-%m-%d')} {trade['buy_price']:,.0f}円")
            print(f"  売り: {trade['sell_date'].strftime('%y-%m-%d')} {trade['sell_price']:,.0f}円")
            print(f"  損益: {profit:,.0f}円")
        
        print(f"\nその他取引の総損益: {total_other_profit:,.0f}円")
    
    # 全体の結果
    total_profit = final_cash - initial_cash
    print(f"\n💰 全体結果:")
    print(f"最終資産: {final_cash:,.0f}円")
    print(f"総損益: {total_profit:,.0f}円")
    print(f"総収益率: {total_profit / initial_cash:.2%}")
    
    # トレーリングストップの効果分析
    if trailing_stop_trades:
        trailing_stop_effectiveness = len([t for t in trailing_stop_trades if t['profit'] > 0]) / len(trailing_stop_trades) * 100
        print(f"\n📊 トレーリングストップ効果:")
        print(f"トレーリングストップ勝率: {trailing_stop_effectiveness:.1f}%")
        
        avg_stop_improvement = np.mean([t['final_stop'] - t['initial_stop'] for t in trailing_stop_trades])
        print(f"平均ストップ改善額: {avg_stop_improvement:,.0f}円")
    
    # 詳細な結果表示
    system.show_results()
    
    print("\n=== トレーリングストップ機能テスト完了 ===")

def compare_with_fixed_stop():
    """固定ストップロスとの比較"""
    print("\n=== 固定ストップロスとの比較 ===")
    
    stock_code = "7203.JP"
    initial_cash = 1_000_000
    
    # 現在の設定（トレーリングストップ有効）
    print("1. トレーリングストップ有効でのバックテスト...")
    system_trailing = TradingSystem(stock_code=stock_code)
    system_trailing.initial_cash = initial_cash
    
    df = system_trailing.prepare_data()
    if df is not None:
        asset_history_trailing, trade_history_trailing, final_cash_trailing = system_trailing.run_simulation()
        if asset_history_trailing is not None:
            trailing_profit = final_cash_trailing - initial_cash
            print(f"トレーリングストップ結果: {trailing_profit:,.0f}円")
        else:
            trailing_profit = 0
    else:
        trailing_profit = 0
    
    # 固定ストップロスでの比較（設定を変更）
    print("\n2. 固定ストップロスでのバックテスト...")
    system_fixed = TradingSystem(stock_code=stock_code)
    system_fixed.initial_cash = initial_cash
    
    # トレーリングストップを無効化（利食いと損切りのみ）
    system_fixed.take_profit_atr_multiple = 3.0
    system_fixed.stop_loss_atr_multiple = 1.5
    system_fixed.take_profit_rate = 0.10
    system_fixed.stop_loss_rate = 0.05
    
    df = system_fixed.prepare_data()
    if df is not None:
        asset_history_fixed, trade_history_fixed, final_cash_fixed = system_fixed.run_simulation()
        if asset_history_fixed is not None:
            fixed_profit = final_cash_fixed - initial_cash
            print(f"固定ストップロス結果: {fixed_profit:,.0f}円")
        else:
            fixed_profit = 0
    else:
        fixed_profit = 0
    
    # 比較結果
    print(f"\n📊 比較結果:")
    print(f"トレーリングストップ: {trailing_profit:,.0f}円")
    print(f"固定ストップロス: {fixed_profit:,.0f}円")
    
    if trailing_profit > fixed_profit:
        improvement = ((trailing_profit - fixed_profit) / abs(fixed_profit)) * 100 if fixed_profit != 0 else 0
        print(f"✅ トレーリングストップの改善: {improvement:.1f}%")
    elif fixed_profit > trailing_profit:
        degradation = ((fixed_profit - trailing_profit) / abs(trailing_profit)) * 100 if trailing_profit != 0 else 0
        print(f"❌ トレーリングストップの劣化: {degradation:.1f}%")
    else:
        print("⚖️ 同じ結果")

if __name__ == "__main__":
    test_trailing_stop()
    compare_with_fixed_stop() 