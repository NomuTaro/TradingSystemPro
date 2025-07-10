#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading System Pro - ウォークフォワード分析テスト

このスクリプトは、ウォークフォワード分析の動作をテストします。
短い期間でテストできるように設定されています。
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

from walk_forward_analysis import WalkForwardAnalyzer
import warnings

warnings.simplefilter('ignore')

def test_walk_forward_analysis():
    """ウォークフォワード分析のテスト"""
    print("=== Trading System Pro - ウォークフォワード分析テスト ===\n")
    
    # テスト設定（短い期間でテスト）
    stock_code = "7203.JP"  # トヨタ自動車
    initial_cash = 1_000_000  # 100万円
    
    print(f"銘柄: {stock_code}")
    print(f"初期資金: {initial_cash:,.0f}円")
    print("="*50)
    
    # ウォークフォワード分析実行
    analyzer = WalkForwardAnalyzer(stock_code=stock_code, initial_cash=initial_cash)
    
    # テスト用に短い期間で実行
    print("注意: テスト用に短い期間で実行します。")
    print("本格的な分析には walk_forward_analysis.py を使用してください。")
    
    # 短い期間でテスト
    results = analyzer.run_walk_forward_analysis(
        training_period_years=1,  # 1年に短縮
        validation_period_months=3,  # 3ヶ月に短縮
        step_months=3,  # 3ヶ月ずつずらす
        n_trials=10  # 少ない試行回数でテスト
    )
    
    if results:
        print(f"\n✅ テスト完了!")
        print(f"分析期間数: {len(results)}")
        
        # 結果のサマリーを表示
        summary = analyzer.analyze_results()
        if summary:
            print(f"\n📊 テスト結果サマリー:")
            print(f"平均総損益: {summary['avg_total_profit']:,.0f}円")
            print(f"平均収益率: {summary['avg_total_return']:.2%}")
            print(f"平均プロフィットファクター: {summary['avg_profit_factor']:.2f}")
            print(f"期間勝率: {summary['period_win_rate']:.1f}%")
        
        # 結果を保存
        analyzer.save_results("test_walk_forward_results.csv")
        
        print(f"\n=== ウォークフォワード分析テスト完了 ===")
    else:
        print(f"\n❌ テストに失敗しました。")

def test_performance_metrics():
    """パフォーマンス指標計算のテスト"""
    print("\n=== パフォーマンス指標計算テスト ===")
    
    # テスト用のダミーデータ
    trade_history = [
        {'type': 'BUY', 'price': 100, 'date': '2023-01-01'},
        {'type': 'SELL', 'price': 110, 'date': '2023-01-10'},
        {'type': 'BUY', 'price': 105, 'date': '2023-01-15'},
        {'type': 'SELL', 'price': 95, 'date': '2023-01-20'},
    ]
    
    asset_history = [
        {'total_value': 1000000, 'date': '2023-01-01'},
        {'total_value': 1010000, 'date': '2023-01-10'},
        {'total_value': 1005000, 'date': '2023-01-15'},
        {'total_value': 995000, 'date': '2023-01-20'},
    ]
    
    analyzer = WalkForwardAnalyzer("7203.JP", 1000000)
    metrics = analyzer.calculate_performance_metrics(trade_history, asset_history, 1000000)
    
    print(f"総損益: {metrics['total_profit']:,.0f}円")
    print(f"収益率: {metrics['total_return']:.2%}")
    print(f"プロフィットファクター: {metrics['profit_factor']:.2f}")
    print(f"最大ドローダウン: {metrics['max_drawdown']:.2%}")
    print(f"勝率: {metrics['win_rate']:.1f}%")
    print(f"取引回数: {metrics['total_trades']}回")

if __name__ == "__main__":
    test_performance_metrics()
    test_walk_forward_analysis() 