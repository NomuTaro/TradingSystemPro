#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading System Pro - Optuna最適化テスト

このスクリプトは、Optunaを使った移動平均線期間の最適化をテストします。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import optuna
except ImportError:
    print("Optunaがインストールされていません。以下のコマンドでインストールしてください:")
    print("pip install optuna")
    sys.exit(1)

import pandas as pd
import numpy as np
from trading_system import TradingSystem
import config
from typing import Dict, Any, Optional
import warnings

warnings.simplefilter('ignore')

def simple_optimization_test():
    """シンプルな最適化テスト"""
    print("=== Trading System Pro - Optuna最適化テスト ===\n")
    
    # 設定
    stock_code = "7203.JP"  # トヨタ自動車
    initial_cash = 1_000_000  # 100万円
    n_trials = 20  # 少ない試行回数でテスト
    
    print(f"銘柄: {stock_code}")
    print(f"初期資金: {initial_cash:,.0f}円")
    print(f"試行回数: {n_trials}")
    print("="*50)
    
    def objective(trial):
        """Optunaの目的関数"""
        # 移動平均線の期間をサンプリング
        sma_short = trial.suggest_int('sma_short', 3, 15, step=1)
        sma_medium = trial.suggest_int('sma_medium', 20, 40, step=1)
        sma_long = trial.suggest_int('sma_long', 50, 100, step=1)
        
        # 制約: 短期 < 中期 < 長期
        if not (sma_short < sma_medium < sma_long):
            return float('-inf')
        
        try:
            # TradingSystemインスタンスを作成
            system = TradingSystem(stock_code=stock_code)
            system.initial_cash = initial_cash
            
            # データを準備
            df = system.prepare_data()
            if df is None:
                return float('-inf')
            
            # 移動平均線を再計算
            df['sma5'] = df['Close'].rolling(window=sma_short).mean()
            df['sma25'] = df['Close'].rolling(window=sma_medium).mean()
            df['sma75'] = df['Close'].rolling(window=sma_long).mean()
            
            # データフレームを更新
            system.df = df
            
            # シミュレーション実行
            asset_history, trade_history, final_cash = system.run_simulation()
            
            if asset_history is None or trade_history is None:
                return float('-inf')
            
            # 総損益を計算
            total_profit = final_cash - initial_cash
            
            print(f"試行 {trial.number}: SMA({sma_short}, {sma_medium}, {sma_long}) -> 損益: {total_profit:,.0f}円")
            
            return total_profit
            
        except Exception as e:
            print(f"エラー in trial {trial.number}: {e}")
            return float('-inf')
    
    # Optunaスタディーを作成
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # 最適化実行
    print("最適化開始...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # 結果表示
    print(f"\n✅ 最適化完了!")
    print(f"最適パラメータ:")
    print(f"  短期移動平均: {study.best_params['sma_short']}日")
    print(f"  中期移動平均: {study.best_params['sma_medium']}日")
    print(f"  長期移動平均: {study.best_params['sma_long']}日")
    print(f"最適総損益: {study.best_value:,.0f}円")
    
    # 最適パラメータでのバックテスト
    print(f"\n📊 最適パラメータでのバックテスト実行...")
    
    system = TradingSystem(stock_code=stock_code)
    system.initial_cash = initial_cash
    
    df = system.prepare_data()
    if df is not None:
        # 移動平均線を再計算
        df['sma5'] = df['Close'].rolling(window=study.best_params['sma_short']).mean()
        df['sma25'] = df['Close'].rolling(window=study.best_params['sma_medium']).mean()
        df['sma75'] = df['Close'].rolling(window=study.best_params['sma_long']).mean()
        
        # データフレームを更新
        system.df = df
        
        # シミュレーション実行
        asset_history, trade_history, final_cash = system.run_simulation()
        
        if asset_history is not None and trade_history is not None:
            # 取引分析
            buy_trades = [t for t in trade_history if t['type'] == 'BUY']
            sell_trades = [t for t in trade_history if t['type'] == 'SELL']
            
            # 損益計算
            total_profit = final_cash - initial_cash
            total_return = total_profit / initial_cash
            
            # 勝率計算
            winning_trades = 0
            total_trades = len(buy_trades)
            
            for i, buy_trade in enumerate(buy_trades):
                if i < len(sell_trades):
                    buy_price = buy_trade['price']
                    sell_price = sell_trades[i]['price']
                    if sell_price > buy_price:
                        winning_trades += 1
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            print(f"\n🎯 最適化バックテスト結果:")
            print(f"最終資産: {final_cash:,.0f}円")
            print(f"総損益: {total_profit:,.0f}円")
            print(f"総収益率: {total_return:.2%}")
            print(f"取引回数: {total_trades}回")
            print(f"勝率: {win_rate:.1f}%")
            
            # 結果を表示
            system.show_results()
    
    print("\n=== 最適化テスト完了 ===")

def compare_with_default():
    """デフォルトパラメータとの比較"""
    print("\n=== デフォルトパラメータとの比較 ===")
    
    stock_code = "7203.JP"
    initial_cash = 1_000_000
    
    # デフォルトパラメータでのバックテスト
    print("1. デフォルトパラメータでのバックテスト...")
    default_system = TradingSystem(stock_code=stock_code)
    default_system.initial_cash = initial_cash
    
    df_default = default_system.prepare_data()
    if df_default is not None:
        asset_history_default, trade_history_default, final_cash_default = default_system.run_simulation()
        if asset_history_default is not None and trade_history_default is not None:
            default_profit = final_cash_default - initial_cash
            
            print(f"デフォルトパラメータ結果:")
            print(f"  最終資産: {final_cash_default:,.0f}円")
            print(f"  総損益: {default_profit:,.0f}円")
            print(f"  取引回数: {len([t for t in trade_history_default if t['type'] == 'BUY'])}回")
        else:
            default_profit = 0
    else:
        default_profit = 0
    
    # 最適化パラメータでのバックテスト（簡易版）
    print("\n2. 最適化パラメータでのバックテスト...")
    
    def quick_objective(trial):
        sma_short = trial.suggest_int('sma_short', 5, 15, step=1)
        sma_medium = trial.suggest_int('sma_medium', 20, 40, step=1)
        sma_long = trial.suggest_int('sma_long', 50, 100, step=1)
        
        if not (sma_short < sma_medium < sma_long):
            return float('-inf')
        
        try:
            system = TradingSystem(stock_code=stock_code)
            system.initial_cash = initial_cash
            
            df = system.prepare_data()
            if df is None:
                return float('-inf')
            
            df['sma5'] = df['Close'].rolling(window=sma_short).mean()
            df['sma25'] = df['Close'].rolling(window=sma_medium).mean()
            df['sma75'] = df['Close'].rolling(window=sma_long).mean()
            
            system.df = df
            
            asset_history, trade_history, final_cash = system.run_simulation()
            
            if asset_history is None or trade_history is None:
                return float('-inf')
            
            return final_cash - initial_cash
            
        except Exception:
            return float('-inf')
    
    study = optuna.create_study(direction='maximize')
    study.optimize(quick_objective, n_trials=10, show_progress_bar=False)
    
    if study.best_value > float('-inf'):
        optimized_profit = study.best_value
        
        print(f"\n📊 比較結果:")
        print(f"デフォルトパラメータ総損益: {default_profit:,.0f}円")
        print(f"最適化パラメータ総損益: {optimized_profit:,.0f}円")
        
        if optimized_profit > default_profit:
            improvement = ((optimized_profit - default_profit) / abs(default_profit)) * 100 if default_profit != 0 else 0
            print(f"✅ 改善率: {improvement:.1f}%")
        else:
            print("❌ デフォルトパラメータの方が良い結果でした。")

if __name__ == "__main__":
    simple_optimization_test()
    compare_with_default() 