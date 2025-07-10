#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading System Pro - ADX機能テスト

このスクリプトは、新しく追加されたADX（Average Directional Index）機能をテストします。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from trading_system import TradingSystem
import config
import warnings

warnings.simplefilter('ignore')

def test_adx_calculation():
    """ADX計算のテスト"""
    print("=== Trading System Pro - ADX機能テスト ===\n")
    
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
    print(f"期間: {df.index[0].date()} ~ {df.index[-1].date()}")
    
    # ADX関連カラムの確認
    adx_columns = ['ADX', 'PLUS_DI', 'MINUS_DI']
    print(f"\n📊 ADX関連カラム:")
    for col in adx_columns:
        if col in df.columns:
            print(f"  ✅ {col}: 利用可能")
            print(f"     範囲: {df[col].min():.2f} ~ {df[col].max():.2f}")
            print(f"     平均: {df[col].mean():.2f}")
        else:
            print(f"  ❌ {col}: 利用不可")
    
    # ADX統計情報
    if 'ADX' in df.columns:
        print(f"\n📈 ADX統計情報:")
        print(f"  平均ADX: {df['ADX'].mean():.2f}")
        print(f"  最大ADX: {df['ADX'].max():.2f}")
        print(f"  最小ADX: {df['ADX'].min():.2f}")
        print(f"  標準偏差: {df['ADX'].std():.2f}")
        
        # ADX強度の分析
        strong_trend_days = len(df[df['ADX'] > 25])
        very_strong_trend_days = len(df[df['ADX'] > 50])
        weak_trend_days = len(df[df['ADX'] < 20])
        
        print(f"\n🎯 ADX強度分析:")
        print(f"  強いトレンド日数 (ADX > 25): {strong_trend_days}日 ({strong_trend_days/len(df)*100:.1f}%)")
        print(f"  非常に強いトレンド日数 (ADX > 50): {very_strong_trend_days}日 ({very_strong_trend_days/len(df)*100:.1f}%)")
        print(f"  弱いトレンド日数 (ADX < 20): {weak_trend_days}日 ({weak_trend_days/len(df)*100:.1f}%)")
    
    # DIクロス分析
    if 'PLUS_DI' in df.columns and 'MINUS_DI' in df.columns:
        print(f"\n🔄 DIクロス分析:")
        
        # ブリッシュクロス（+DIが-DIを上向きにクロス）
        bullish_crosses = 0
        bearish_crosses = 0
        
        for i in range(1, len(df)):
            prev_plus_di = df['PLUS_DI'].iloc[i-1]
            prev_minus_di = df['MINUS_DI'].iloc[i-1]
            curr_plus_di = df['PLUS_DI'].iloc[i]
            curr_minus_di = df['MINUS_DI'].iloc[i]
            
            # ブリッシュクロス
            if prev_plus_di < prev_minus_di and curr_plus_di > curr_minus_di:
                bullish_crosses += 1
            
            # ベアリッシュクロス
            if prev_minus_di < prev_plus_di and curr_minus_di > curr_plus_di:
                bearish_crosses += 1
        
        print(f"  ブリッシュクロス回数: {bullish_crosses}回")
        print(f"  ベアリッシュクロス回数: {bearish_crosses}回")
    
    return df

def analyze_adx_signals(df):
    """ADXシグナルの分析"""
    print(f"\n📊 ADXシグナル分析")
    print("="*50)
    
    if 'ADX' not in df.columns or 'PLUS_DI' not in df.columns or 'MINUS_DI' not in df.columns:
        print("ADX関連データが利用できません。")
        return
    
    # シグナル条件の確認
    signals = []
    
    for i in range(len(df)):
        adx = df['ADX'].iloc[i]
        plus_di = df['PLUS_DI'].iloc[i]
        minus_di = df['MINUS_DI'].iloc[i]
        date = df.index[i]
        
        # 強い上昇トレンド
        if adx > 25 and plus_di > minus_di:
            signals.append({
                'date': date,
                'type': 'strong_uptrend',
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di
            })
        
        # 強い下降トレンド
        elif adx > 25 and minus_di > plus_di:
            signals.append({
                'date': date,
                'type': 'strong_downtrend',
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di
            })
    
    # 結果表示
    uptrend_signals = [s for s in signals if s['type'] == 'strong_uptrend']
    downtrend_signals = [s for s in signals if s['type'] == 'strong_downtrend']
    
    print(f"強い上昇トレンドシグナル: {len(uptrend_signals)}回")
    print(f"強い下降トレンドシグナル: {len(downtrend_signals)}回")
    
    if uptrend_signals:
        print(f"\n📈 強い上昇トレンド期間:")
        for signal in uptrend_signals[-5:]:  # 最新5件
            print(f"  {signal['date'].strftime('%Y-%m-%d')}: ADX={signal['adx']:.1f}, +DI={signal['plus_di']:.1f}, -DI={signal['minus_di']:.1f}")
    
    if downtrend_signals:
        print(f"\n📉 強い下降トレンド期間:")
        for signal in downtrend_signals[-5:]:  # 最新5件
            print(f"  {signal['date'].strftime('%Y-%m-%d')}: ADX={signal['adx']:.1f}, +DI={signal['plus_di']:.1f}, -DI={signal['minus_di']:.1f}")

def plot_adx_analysis(df):
    """ADX分析の可視化"""
    print(f"\n📊 ADX分析の可視化")
    print("="*50)
    
    if 'ADX' not in df.columns or 'PLUS_DI' not in df.columns or 'MINUS_DI' not in df.columns:
        print("ADX関連データが利用できません。")
        return
    
    # プロット設定
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'ADX Analysis - {df.index[0].date()} to {df.index[-1].date()}', fontsize=16)
    
    # 1. ADX推移
    axes[0, 0].plot(df.index, df['ADX'], 'black', linewidth=1.5, label='ADX')
    axes[0, 0].axhline(y=25, color='orange', linestyle='--', alpha=0.7, label='Strong Trend (25)')
    axes[0, 0].axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Very Strong Trend (50)')
    axes[0, 0].set_title('ADX Trend Strength')
    axes[0, 0].set_ylabel('ADX')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. +DI vs -DI
    axes[0, 1].plot(df.index, df['PLUS_DI'], 'green', linewidth=1.5, label='+DI')
    axes[0, 1].plot(df.index, df['MINUS_DI'], 'red', linewidth=1.5, label='-DI')
    axes[0, 1].set_title('Directional Indicators')
    axes[0, 1].set_ylabel('DI Values')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. ADX分布
    axes[1, 0].hist(df['ADX'].dropna(), bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].axvline(x=25, color='orange', linestyle='--', alpha=0.7, label='Strong Trend')
    axes[1, 0].axvline(x=50, color='red', linestyle='--', alpha=0.7, label='Very Strong Trend')
    axes[1, 0].set_title('ADX Distribution')
    axes[1, 0].set_xlabel('ADX')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. ADX vs 価格変化率
    price_change = df['Close'].pct_change() * 100
    axes[1, 1].scatter(df['ADX'], price_change, alpha=0.6, s=20)
    axes[1, 1].set_title('ADX vs Price Change')
    axes[1, 1].set_xlabel('ADX')
    axes[1, 1].set_ylabel('Price Change (%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def test_adx_in_trading():
    """ADX機能付きトレーディングテスト"""
    print(f"\n🚀 ADX機能付きトレーディングテスト")
    print("="*50)
    
    stock_code = "7203.JP"
    initial_cash = 1_000_000
    
    # TradingSystemインスタンスを作成
    system = TradingSystem(stock_code=stock_code)
    system.initial_cash = initial_cash
    
    # データを準備
    df = system.prepare_data()
    if df is None:
        print("データ取得に失敗しました。")
        return
    
    # シミュレーション実行
    print("ADX機能付きシミュレーション実行中...")
    asset_history, trade_history, final_cash = system.run_simulation()
    
    if asset_history is None or trade_history is None:
        print("シミュレーションに失敗しました。")
        return
    
    # 結果表示
    total_profit = final_cash - initial_cash
    total_return = total_profit / initial_cash
    
    print(f"\n💰 トレーディング結果:")
    print(f"最終資産: {final_cash:,.0f}円")
    print(f"総損益: {total_profit:,.0f}円")
    print(f"総収益率: {total_return:.2%}")
    print(f"取引回数: {len([t for t in trade_history if t['type'] == 'BUY'])}回")
    
    # ADXシグナルによる取引の分析
    adx_trades = []
    for i, trade in enumerate(trade_history):
        if trade['type'] == 'BUY' and i + 1 < len(trade_history):
            sell_trade = trade_history[i + 1]
            buy_date = trade['date']
            sell_date = sell_trade['date']
            
            # 取引期間のADXデータを取得
            trade_period = df[(df.index >= buy_date) & (df.index <= sell_date)]
            if len(trade_period) > 0:
                avg_adx = trade_period['ADX'].mean()
                max_adx = trade_period['ADX'].max()
                strong_trend_days = len(trade_period[trade_period['ADX'] > 25])
                
                adx_trades.append({
                    'buy_date': buy_date,
                    'sell_date': sell_date,
                    'profit': sell_trade['proceeds'] - trade['cost'],
                    'avg_adx': avg_adx,
                    'max_adx': max_adx,
                    'strong_trend_days': strong_trend_days,
                    'total_days': len(trade_period)
                })
    
    if adx_trades:
        print(f"\n📊 ADX取引分析:")
        print(f"分析対象取引数: {len(adx_trades)}回")
        
        # 強いトレンド期間の取引
        strong_trend_trades = [t for t in adx_trades if t['strong_trend_days'] > 0]
        weak_trend_trades = [t for t in adx_trades if t['strong_trend_days'] == 0]
        
        if strong_trend_trades:
            strong_profits = [t['profit'] for t in strong_trend_trades]
            avg_strong_profit = np.mean(strong_profits)
            strong_win_rate = len([p for p in strong_profits if p > 0]) / len(strong_profits) * 100
            
            print(f"強いトレンド期間の取引: {len(strong_trend_trades)}回")
            print(f"  平均損益: {avg_strong_profit:,.0f}円")
            print(f"  勝率: {strong_win_rate:.1f}%")
        
        if weak_trend_trades:
            weak_profits = [t['profit'] for t in weak_trend_trades]
            avg_weak_profit = np.mean(weak_profits)
            weak_win_rate = len([p for p in weak_profits if p > 0]) / len(weak_profits) * 100
            
            print(f"弱いトレンド期間の取引: {len(weak_trend_trades)}回")
            print(f"  平均損益: {avg_weak_profit:,.0f}円")
            print(f"  勝率: {weak_win_rate:.1f}%")
    
    # 詳細な結果表示
    system.show_results()

if __name__ == "__main__":
    # ADX計算テスト
    df = test_adx_calculation()
    
    if df is not None:
        # ADXシグナル分析
        analyze_adx_signals(df)
        
        # ADX可視化
        plot_adx_analysis(df)
        
        # ADX機能付きトレーディングテスト
        test_adx_in_trading()
    
    print("\n=== ADX機能テスト完了 ===") 