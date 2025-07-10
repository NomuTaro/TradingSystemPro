#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading System Pro - Optunaパラメータ最適化サンプル

このスクリプトは、TradingSystemProの移動平均線期間をOptunaを使って最適化します。
総損益を最大化するように移動平均線の期間（5, 25, 75）を調整します。
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
import matplotlib.pyplot as plt
from trading_system import TradingSystem
import config
from typing import Dict, Any, Tuple, Optional
import warnings

warnings.simplefilter('ignore')

class TradingSystemOptimizer:
    """TradingSystemのパラメータ最適化クラス"""
    
    def __init__(self, stock_code: Optional[str] = None, initial_cash: Optional[float] = None):
        """
        初期化
        
        Args:
            stock_code (str): 銘柄コード
            initial_cash (float): 初期資金
        """
        self.stock_code = stock_code or config.CONFIG['DEFAULT_SYMBOL']
        self.initial_cash = initial_cash or config.CONFIG['INITIAL_CASH']
        self.best_params = None
        self.best_value = float('-inf')
        self.optimization_history = []
        
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optunaの目的関数
        
        Args:
            trial (optuna.Trial): Optunaのトライアルオブジェクト
            
        Returns:
            float: 総損益（最大化対象）
        """
        # 移動平均線の期間をサンプリング
        sma_short = trial.suggest_int('sma_short', 3, 20, step=1)      # 短期移動平均（3-20日）
        sma_medium = trial.suggest_int('sma_medium', 15, 50, step=1)   # 中期移動平均（15-50日）
        sma_long = trial.suggest_int('sma_long', 40, 200, step=1)      # 長期移動平均（40-200日）
        
        # 制約: 短期 < 中期 < 長期
        if not (sma_short < sma_medium < sma_long):
            return float('-inf')  # 制約違反の場合は極小値を返す
        
        try:
            # TradingSystemインスタンスを作成
            system = TradingSystem(stock_code=self.stock_code)
            
            # 設定を更新
            system.initial_cash = self.initial_cash
            
            # データを準備
            df = system.prepare_data()
            if df is None:
                return float('-inf')
            
            # 移動平均線を再計算（カスタム期間）
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
            total_profit = final_cash - self.initial_cash
            
            # 最適化履歴に記録
            self.optimization_history.append({
                'trial': trial.number,
                'sma_short': sma_short,
                'sma_medium': sma_medium,
                'sma_long': sma_long,
                'total_profit': total_profit,
                'final_cash': final_cash,
                'trade_count': len([t for t in trade_history if t['type'] == 'BUY'])
            })
            
            # 最良結果を更新
            if total_profit > self.best_value:
                self.best_value = total_profit
                self.best_params = {
                    'sma_short': sma_short,
                    'sma_medium': sma_medium,
                    'sma_long': sma_long
                }
            
            return total_profit
            
        except Exception as e:
            print(f"エラー in trial {trial.number}: {e}")
            return float('-inf')
    
    def optimize(self, n_trials: int = 100, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        パラメータ最適化を実行
        
        Args:
            n_trials (int): 試行回数
            timeout (int): タイムアウト（秒）
            
        Returns:
            Dict[str, Any]: 最適化結果
        """
        print(f"🚀 Optuna最適化開始")
        print(f"銘柄: {self.stock_code}")
        print(f"初期資金: {self.initial_cash:,.0f}円")
        print(f"試行回数: {n_trials}")
        print("="*50)
        
        # Optunaスタディーを作成
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # 最適化実行
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # 結果を取得
        best_params = study.best_params
        best_value = study.best_value
        
        print(f"\n✅ 最適化完了!")
        print(f"最適パラメータ:")
        print(f"  短期移動平均: {best_params['sma_short']}日")
        print(f"  中期移動平均: {best_params['sma_medium']}日")
        print(f"  長期移動平均: {best_params['sma_long']}日")
        print(f"最適総損益: {best_value:,.0f}円")
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'study': study,
            'optimization_history': self.optimization_history
        }
    
    def run_optimized_backtest(self, params: Dict[str, int]) -> Dict[str, Any]:
        """
        最適化されたパラメータでバックテストを実行
        
        Args:
            params (Dict[str, int]): 最適化されたパラメータ
            
        Returns:
            Dict[str, Any]: バックテスト結果
        """
        print(f"\n📊 最適パラメータでのバックテスト実行...")
        
        # TradingSystemインスタンスを作成
        system = TradingSystem(stock_code=self.stock_code)
        system.initial_cash = self.initial_cash
        
        # データを準備
        df = system.prepare_data()
        if df is None:
            return {}
        
        # 移動平均線を再計算
        df['sma5'] = df['Close'].rolling(window=params['sma_short']).mean()
        df['sma25'] = df['Close'].rolling(window=params['sma_medium']).mean()
        df['sma75'] = df['Close'].rolling(window=params['sma_long']).mean()
        
        # データフレームを更新
        system.df = df
        
        # シミュレーション実行
        asset_history, trade_history, final_cash = system.run_simulation()
        
        if asset_history is None or trade_history is None:
            return {}
        
        # 取引分析
        buy_trades = [t for t in trade_history if t['type'] == 'BUY']
        sell_trades = [t for t in trade_history if t['type'] == 'SELL']
        
        # 損益計算
        total_profit = final_cash - self.initial_cash
        total_return = total_profit / self.initial_cash
        
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
        
        # 結果を表示
        system.show_results()
        
        return {
            'final_cash': final_cash,
            'total_profit': total_profit,
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'asset_history': asset_history,
            'trade_history': trade_history
        }
    
    def plot_optimization_results(self, history: list):
        """
        最適化結果を可視化
        
        Args:
            history (list): 最適化履歴
        """
        if not history:
            print("最適化履歴がありません。")
            return
        
        df_history = pd.DataFrame(history)
        
        # プロット設定
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Optuna最適化結果 - {self.stock_code}', fontsize=16)
        
        # 1. 総損益の推移
        axes[0, 0].plot(df_history['trial'], df_history['total_profit'], 'b-', alpha=0.7)
        axes[0, 0].set_title('総損益の推移')
        axes[0, 0].set_xlabel('試行回数')
        axes[0, 0].set_ylabel('総損益 (円)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. パラメータの分布
        axes[0, 1].scatter(df_history['sma_short'], df_history['total_profit'], 
                          alpha=0.6, label='短期MA')
        axes[0, 1].scatter(df_history['sma_medium'], df_history['total_profit'], 
                          alpha=0.6, label='中期MA')
        axes[0, 1].scatter(df_history['sma_long'], df_history['total_profit'], 
                          alpha=0.6, label='長期MA')
        axes[0, 1].set_title('移動平均期間と総損益の関係')
        axes[0, 1].set_xlabel('移動平均期間 (日)')
        axes[0, 1].set_ylabel('総損益 (円)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 取引回数の分布
        axes[1, 0].hist(df_history['trade_count'], bins=20, alpha=0.7, color='green')
        axes[1, 0].set_title('取引回数の分布')
        axes[1, 0].set_xlabel('取引回数')
        axes[1, 0].set_ylabel('頻度')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 3D散布図（短期MA vs 中期MA vs 総損益）
        try:
            from mpl_toolkits.mplot3d import Axes3D
            # 3Dプロット用のサブプロットを作成
            ax3d = fig.add_subplot(2, 2, 4, projection='3d')
            scatter = ax3d.scatter(df_history['sma_short'], df_history['sma_medium'], 
                                  df_history['total_profit'], 
                                  c=df_history['total_profit'], cmap='viridis')
            ax3d.set_title('短期MA vs 中期MA vs 総損益')
            ax3d.set_xlabel('短期MA (日)')
            ax3d.set_ylabel('中期MA (日)')
            # set_zlabelはAxes3D型のときのみ呼び出す
            if isinstance(ax3d, Axes3D):
                ax3d.set_zlabel('総損益 (円)')
            plt.colorbar(scatter, ax=ax3d)
        except ImportError:
            # 3Dプロットが利用できない場合は2D散布図で代替
            ax3d = fig.add_subplot(2, 2, 4)
            scatter = ax3d.scatter(df_history['sma_short'], df_history['sma_medium'], 
                                  c=df_history['total_profit'], cmap='viridis')
            ax3d.set_title('短期MA vs 中期MA vs 総損益')
            ax3d.set_xlabel('短期MA (日)')
            ax3d.set_ylabel('中期MA (日)')
            plt.colorbar(scatter, ax=ax3d)
        
        plt.tight_layout()
        plt.show()
        
        # 統計情報を表示
        print(f"\n📈 最適化統計情報:")
        print(f"試行回数: {len(history)}")
        print(f"平均総損益: {df_history['total_profit'].mean():,.0f}円")
        print(f"最大総損益: {df_history['total_profit'].max():,.0f}円")
        print(f"最小総損益: {df_history['total_profit'].min():,.0f}円")
        print(f"標準偏差: {df_history['total_profit'].std():,.0f}円")
        print(f"平均取引回数: {df_history['trade_count'].mean():.1f}回")


def main():
    """メイン実行関数"""
    print("=== Trading System Pro - Optuna最適化デモ ===\n")
    
    # 1. 最適化設定
    stock_code = "7203.JP"  # トヨタ自動車
    initial_cash = 1_000_000  # 100万円
    n_trials = 50  # 試行回数（時間短縮のため50回）
    
    # 2. 最適化実行
    optimizer = TradingSystemOptimizer(
        stock_code=stock_code,
        initial_cash=initial_cash
    )
    
    # 最適化実行
    results = optimizer.optimize(n_trials=n_trials)
    
    # 3. 最適パラメータでのバックテスト
    if results['best_params']:
        backtest_results = optimizer.run_optimized_backtest(results['best_params'])
        
        if backtest_results:
            print(f"\n🎯 最適化バックテスト結果サマリー:")
            print(f"最終資産: {backtest_results['final_cash']:,.0f}円")
            print(f"総損益: {backtest_results['total_profit']:,.0f}円")
            print(f"総収益率: {backtest_results['total_return']:.2%}")
            print(f"取引回数: {backtest_results['total_trades']}回")
            print(f"勝率: {backtest_results['win_rate']:.1f}%")
    
    # 4. 最適化結果の可視化
    if optimizer.optimization_history:
        optimizer.plot_optimization_results(optimizer.optimization_history)
    
    print("\n=== Optuna最適化デモ完了 ===")


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
    
    # 最適化パラメータでのバックテスト
    print("\n2. 最適化パラメータでのバックテスト...")
    optimizer = TradingSystemOptimizer(stock_code=stock_code, initial_cash=initial_cash)
    results = optimizer.optimize(n_trials=30)  # 短時間で比較
    
    if results['best_params']:
        optimized_results = optimizer.run_optimized_backtest(results['best_params'])
        
        if optimized_results and 'total_profit' in optimized_results:
            optimized_profit = optimized_results['total_profit']
            
            print(f"\n📊 比較結果:")
            print(f"デフォルトパラメータ総損益: {default_profit:,.0f}円")
            print(f"最適化パラメータ総損益: {optimized_profit:,.0f}円")
            
            if optimized_profit > default_profit:
                improvement = ((optimized_profit - default_profit) / abs(default_profit)) * 100
                print(f"✅ 改善率: {improvement:.1f}%")
            else:
                print("❌ デフォルトパラメータの方が良い結果でした。")


if __name__ == "__main__":
    main()
    compare_with_default()
