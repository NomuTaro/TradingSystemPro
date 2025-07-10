#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading System Pro - ウォークフォワード分析

このスクリプトは、ウォークフォワード分析を実行します。
学習期間を2年、検証期間を6ヶ月とし、これを6ヶ月ずつずらしながら過去データ全体で繰り返し実行します。
各ループで、学習期間のデータを使ってパラメータを最適化し、そのパラメータで検証期間のパフォーマンスを記録します。
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
from datetime import datetime, timedelta
from trading_system import TradingSystem
import config
from typing import Dict, Any, List, Tuple, Optional
import warnings
from tqdm import tqdm
from pandas._libs.tslibs.nattype import NaTType

warnings.simplefilter('ignore')

def safe_date_str(x):
    if isinstance(x, (pd.Series, pd.Index, pd.DataFrame)):
        return ''
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

def safe_date_add(x, td):
    try:
        ts = pd.Timestamp(x)
        if isinstance(ts, NaTType) or pd.isna(ts):
            return ts
        return ts + td
    except Exception:
        return x

class WalkForwardAnalyzer:
    """ウォークフォワード分析クラス"""
    
    def __init__(self, stock_code: str, initial_cash: float = 1_000_000):
        """
        初期化
        
        Args:
            stock_code (str): 銘柄コード
            initial_cash (float): 初期資金
        """
        self.stock_code = stock_code
        self.initial_cash = initial_cash
        self.results = []
        self.optimization_history = []
        
    def calculate_performance_metrics(self, trade_history: List[Dict[str, Any]], 
                                    asset_history: List[Any], 
                                    initial_cash: float) -> Dict[str, Any]:
        """
        パフォーマンス指標を計算
        
        Args:
            trade_history (List[Dict]): 取引履歴
            asset_history (List[Dict]): 資産履歴
            initial_cash (float): 初期資金
            
        Returns:
            Dict[str, float]: パフォーマンス指標
        """
        if not trade_history or not asset_history:
            return {
                'total_profit': 0.0,
                'total_return': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0
            }
        
        # 取引分析
        buy_trades = [t for t in trade_history if t['type'] == 'BUY']
        sell_trades = [t for t in trade_history if t['type'] == 'SELL']
        
        # 損益計算
        final_cash = float(asset_history[-1]['total_value']) if asset_history else float(initial_cash)
        total_profit = final_cash - initial_cash
        total_return = total_profit / initial_cash if initial_cash > 0 else 0.0
        
        # 個別取引の損益計算
        trade_pnl = []
        for i, buy_trade in enumerate(buy_trades):
            if i < len(sell_trades):
                buy_price = buy_trade['price']
                sell_price = sell_trades[i]['price']
                pnl = sell_price - buy_price
                trade_pnl.append(pnl)
        
        # プロフィットファクター計算
        if trade_pnl:
            gross_profit = sum([pnl for pnl in trade_pnl if pnl > 0])
            gross_loss = abs(sum([pnl for pnl in trade_pnl if pnl < 0]))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            profit_factor = 0.0
        
        # 最大ドローダウン計算
        max_drawdown = 0.0
        if asset_history:
            peak = asset_history[0]['total_value']
            for asset in asset_history:
                if asset['total_value'] > peak:
                    peak = asset['total_value']
                drawdown = (peak - asset['total_value']) / peak if peak > 0 else 0.0
                max_drawdown = max(max_drawdown, drawdown)
        
        # シャープレシオ計算（簡易版）
        if len(asset_history) > 1:
            returns = []
            for i in range(1, len(asset_history)):
                prev_value = asset_history[i-1]['total_value']
                curr_value = asset_history[i]['total_value']
                if prev_value > 0:
                    daily_return = (curr_value - prev_value) / prev_value
                    returns.append(daily_return)
            
            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = avg_return / std_return if std_return > 0 else 0.0
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # 勝率計算
        winning_trades = len([pnl for pnl in trade_pnl if pnl > 0])
        losing_trades = len([pnl for pnl in trade_pnl if pnl < 0])
        total_trades = len(trade_pnl)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        return {
            'total_profit': total_profit,
            'total_return': total_return,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades
        }
    
    def optimize_parameters(self, df_train: pd.DataFrame, n_trials: int = 50) -> Dict[str, Any]:
        """
        学習期間のデータでパラメータを最適化
        
        Args:
            df_train (pd.DataFrame): 学習期間のデータ
            n_trials (int): 最適化試行回数
            
        Returns:
            Dict[str, int]: 最適化されたパラメータ
        """
        def objective(trial):
            # 移動平均線の期間をサンプリング
            sma_short = trial.suggest_int('sma_short', 3, 20, step=1)
            sma_medium = trial.suggest_int('sma_medium', 15, 50, step=1)
            sma_long = trial.suggest_int('sma_long', 40, 200, step=1)
            
            # 制約: 短期 < 中期 < 長期
            if not (sma_short < sma_medium < sma_long):
                return float('-inf')
            
            try:
                # データフレームのコピーを作成
                df_copy = df_train.copy()
                
                # 移動平均線を再計算
                df_copy['sma5'] = df_copy['Close'].rolling(window=sma_short).mean()
                df_copy['sma25'] = df_copy['Close'].rolling(window=sma_medium).mean()
                df_copy['sma75'] = df_copy['Close'].rolling(window=sma_long).mean()
                
                # TradingSystemインスタンスを作成
                system = TradingSystem(stock_code=self.stock_code)
                system.initial_cash = self.initial_cash
                system.df = df_copy
                
                # シミュレーション実行
                asset_history, trade_history, final_cash = system.run_simulation()
                
                if asset_history is None or trade_history is None:
                    return float('-inf')
                
                # 総損益を計算
                total_profit = final_cash - self.initial_cash
                
                return total_profit
                
            except Exception:
                return float('-inf')
        
        # Optunaスタディーを作成
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # 最適化実行
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        return study.best_params
    
    def run_walk_forward_analysis(self, 
                                 training_period_years: int = 2,
                                 validation_period_months: int = 6,
                                 step_months: int = 6,
                                 n_trials: int = 30) -> List[Dict[str, Any]]:
        """
        ウォークフォワード分析を実行
        
        Args:
            training_period_years (int): 学習期間（年）
            validation_period_months (int): 検証期間（月）
            step_months (int): ステップ期間（月）
            n_trials (int): 最適化試行回数
            
        Returns:
            List[Dict[str, Any]]: 分析結果
        """
        print(f"🚀 ウォークフォワード分析開始")
        print(f"銘柄: {self.stock_code}")
        print(f"初期資金: {self.initial_cash:,.0f}円")
        print(f"学習期間: {training_period_years}年")
        print(f"検証期間: {validation_period_months}ヶ月")
        print(f"ステップ期間: {step_months}ヶ月")
        print("="*60)
        
        # データを取得（長期間のデータを取得）
        system = TradingSystem(stock_code=self.stock_code)
        # ウォークフォワード分析用に長期間のデータを取得
        system.data_period_days = 1460  # 約4年分
        df = system.prepare_data()
        
        if df is None or df.empty:
            print("❌ データの取得に失敗しました。")
            return []
        
        # 日付範囲を取得
        start_date = df.index.min()
        end_date = df.index.max()
        # 型安全な日付出力
        print(f"データ期間: {safe_date_str(start_date)} ～ {safe_date_str(end_date)}")
        
        # ウォークフォワード期間を計算
        training_days = training_period_years * 365
        validation_days = validation_period_months * 30
        step_days = step_months * 30
        
        # 分析開始日を設定（学習期間分のデータが必要）
        analysis_start = safe_date_add(start_date, timedelta(days=training_days))
        
        results = []
        current_date = analysis_start
        
        # プログレスバー用の総ループ数を計算
        total_loops = 0
        temp_date = current_date
        while True:
            temp_next = safe_date_add(temp_date, timedelta(days=validation_days))
            # 型安全な比較
            def is_invalid_date(x):
                if isinstance(x, (pd.Series, pd.Index, pd.DataFrame)):
                    return True
                if isinstance(x, NaTType):
                    return True
                return bool(pd.isna(x))
            if is_invalid_date(temp_next) or is_invalid_date(end_date):
                break
            if not (isinstance(temp_next, pd.Timestamp) and isinstance(end_date, pd.Timestamp)):
                break
            if temp_next > end_date:
                break
            total_loops += 1
            temp_date = safe_date_add(temp_date, timedelta(days=step_days))
        
        print(f"予想ループ数: {total_loops}")
        print("="*60)
        
        with tqdm(total=total_loops, desc="ウォークフォワード分析") as pbar:
            while True:
                next_val = safe_date_add(current_date, timedelta(days=validation_days))
                if is_invalid_date(next_val) or is_invalid_date(end_date):
                    break
                if not (isinstance(next_val, pd.Timestamp) and isinstance(end_date, pd.Timestamp)):
                    break
                if next_val > end_date:
                    break
                # 学習期間と検証期間を設定
                train_start = safe_date_add(current_date, timedelta(days=-training_days))
                train_end = current_date
                val_start = current_date
                val_end = safe_date_add(current_date, timedelta(days=validation_days))
                
                print(f"\n�� 期間 {len(results) + 1}:")
                print(f"  学習期間: {safe_date_str(train_start)} ～ {safe_date_str(train_end)}")
                print(f"  検証期間: {safe_date_str(val_start)} ～ {safe_date_str(val_end)}")
                
                # 学習期間のデータを抽出
                df_train = df[(df.index >= train_start) & (df.index <= train_end)].copy()
                if not isinstance(df_train, pd.DataFrame):
                    df_train = pd.DataFrame(df_train)
                
                if len(df_train) < 100:  # 最小データ数チェック
                    print(f"  ⚠️  学習データが不足しています（{len(df_train)}行）。スキップします。")
                    current_date = safe_date_add(current_date, timedelta(days=step_days))
                    pbar.update(1)
                    continue
                
                # パラメータ最適化
                print(f"  🔧 パラメータ最適化中...")
                best_params = self.optimize_parameters(df_train, n_trials=n_trials)
                
                if not best_params:
                    print(f"  ❌ 最適化に失敗しました。スキップします。")
                    current_date = safe_date_add(current_date, timedelta(days=step_days))
                    pbar.update(1)
                    continue
                
                print(f"  ✅ 最適パラメータ: SMA({best_params['sma_short']}, {best_params['sma_medium']}, {best_params['sma_long']})")
                
                # 検証期間のデータを抽出
                df_val = df[(df.index >= val_start) & (df.index <= val_end)].copy()
                if not isinstance(df_val, pd.DataFrame):
                    df_val = pd.DataFrame(df_val)
                
                if len(df_val) < 30:  # 最小データ数チェック
                    print(f"  ⚠️  検証データが不足しています（{len(df_val)}行）。スキップします。")
                    current_date = safe_date_add(current_date, timedelta(days=step_days))
                    pbar.update(1)
                    continue
                
                # 最適化されたパラメータで移動平均線を計算
                if 'Close' in df_val.columns:
                    df_val['sma5'] = pd.Series(df_val['Close']).rolling(window=best_params['sma_short']).mean()
                    df_val['sma25'] = pd.Series(df_val['Close']).rolling(window=best_params['sma_medium']).mean()
                    df_val['sma75'] = pd.Series(df_val['Close']).rolling(window=best_params['sma_long']).mean()
                
                # 検証期間でバックテスト実行
                print(f"  📈 検証期間バックテスト実行中...")
                system_val = TradingSystem(stock_code=self.stock_code)
                system_val.initial_cash = self.initial_cash
                if isinstance(df_val, pd.DataFrame):
                    system_val.df = df_val
                else:
                    system_val.df = pd.DataFrame(df_val)
                
                asset_history, trade_history, final_cash = system_val.run_simulation()
                
                if asset_history is None or trade_history is None:
                    print(f"  ❌ バックテストに失敗しました。スキップします。")
                    current_date = safe_date_add(current_date, timedelta(days=step_days))
                    pbar.update(1)
                    continue
                
                # パフォーマンス指標を計算
                metrics = self.calculate_performance_metrics(trade_history, asset_history, self.initial_cash)
                
                # 結果を記録
                result = {
                    'period': len(results) + 1,
                    'train_start': train_start,
                    'train_end': train_end,
                    'val_start': val_start,
                    'val_end': val_end,
                    'best_params': best_params,
                    'final_cash': final_cash,
                    **metrics
                }
                
                results.append(result)
                
                print(f"  📊 検証結果:")
                print(f"    総損益: {metrics['total_profit']:,.0f}円")
                print(f"    収益率: {metrics['total_return']:.2%}")
                print(f"    プロフィットファクター: {metrics['profit_factor']:.2f}")
                print(f"    最大ドローダウン: {metrics['max_drawdown']:.2%}")
                print(f"    勝率: {metrics['win_rate']:.1f}%")
                print(f"    取引回数: {metrics['total_trades']}回")
                
                # 次の期間に移動
                current_date = safe_date_add(current_date, timedelta(days=step_days))
                pbar.update(1)
        
        self.results = results
        print(f"\n✅ ウォークフォワード分析完了!")
        print(f"総分析期間数: {len(results)}")
        
        return results
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        分析結果を集計・分析
        
        Returns:
            Dict[str, Any]: 分析結果サマリー
        """
        if not self.results:
            print("❌ 分析結果がありません。")
            return {}
        
        df_results = pd.DataFrame(self.results)
        
        # 基本統計
        summary = {
            'total_periods': len(self.results),
            'avg_total_profit': df_results['total_profit'].mean(),
            'avg_total_return': df_results['total_return'].mean(),
            'avg_profit_factor': df_results['profit_factor'].mean(),
            'avg_max_drawdown': df_results['max_drawdown'].mean(),
            'avg_sharpe_ratio': df_results['sharpe_ratio'].mean(),
            'avg_win_rate': df_results['win_rate'].mean(),
            'avg_total_trades': df_results['total_trades'].mean(),
            'profitable_periods': len(df_results[df_results['total_profit'] > 0]),
            'losing_periods': len(df_results[df_results['total_profit'] < 0]),
            'best_period': df_results.loc[df_results['total_profit'].idxmax()],
            'worst_period': df_results.loc[df_results['total_profit'].idxmin()],
            'std_total_profit': df_results['total_profit'].std(),
            'std_total_return': df_results['total_return'].std()
        }
        
        # 勝率計算
        summary['period_win_rate'] = (summary['profitable_periods'] / summary['total_periods']) * 100
        
        return summary
    
    def plot_results(self):
        """分析結果を可視化"""
        if not self.results:
            print("❌ 分析結果がありません。")
            return
        
        df_results = pd.DataFrame(self.results)
        
        # プロット設定
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'ウォークフォワード分析結果 - {self.stock_code}', fontsize=16)
        
        # 1. 総損益の推移
        axes[0, 0].plot(df_results['period'], df_results['total_profit'], 'b-o', alpha=0.7)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('総損益の推移')
        axes[0, 0].set_xlabel('分析期間')
        axes[0, 0].set_ylabel('総損益 (円)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 収益率の推移
        axes[0, 1].plot(df_results['period'], df_results['total_return'] * 100, 'g-o', alpha=0.7)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('収益率の推移')
        axes[0, 1].set_xlabel('分析期間')
        axes[0, 1].set_ylabel('収益率 (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. プロフィットファクターの推移
        axes[1, 0].plot(df_results['period'], df_results['profit_factor'], 'orange', marker='o', alpha=0.7)
        axes[1, 0].axhline(y=1, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('プロフィットファクターの推移')
        axes[1, 0].set_xlabel('分析期間')
        axes[1, 0].set_ylabel('プロフィットファクター')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 最大ドローダウンの推移
        axes[1, 1].plot(df_results['period'], df_results['max_drawdown'] * 100, 'r-o', alpha=0.7)
        axes[1, 1].set_title('最大ドローダウンの推移')
        axes[1, 1].set_xlabel('分析期間')
        axes[1, 1].set_ylabel('最大ドローダウン (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. 勝率の推移
        axes[2, 0].plot(df_results['period'], df_results['win_rate'], 'purple', marker='o', alpha=0.7)
        axes[2, 0].set_title('勝率の推移')
        axes[2, 0].set_xlabel('分析期間')
        axes[2, 0].set_ylabel('勝率 (%)')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. 取引回数の推移
        axes[2, 1].plot(df_results['period'], df_results['total_trades'], 'brown', marker='o', alpha=0.7)
        axes[2, 1].set_title('取引回数の推移')
        axes[2, 1].set_xlabel('分析期間')
        axes[2, 1].set_ylabel('取引回数')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 統計情報を表示
        summary = self.analyze_results()
        if summary:
            print(f"\n📈 ウォークフォワード分析サマリー:")
            print(f"総分析期間数: {summary['total_periods']}")
            print(f"平均総損益: {summary['avg_total_profit']:,.0f}円")
            print(f"平均収益率: {summary['avg_total_return']:.2%}")
            print(f"平均プロフィットファクター: {summary['avg_profit_factor']:.2f}")
            print(f"平均最大ドローダウン: {summary['avg_max_drawdown']:.2%}")
            print(f"平均勝率: {summary['avg_win_rate']:.1f}%")
            print(f"期間勝率: {summary['period_win_rate']:.1f}%")
            print(f"平均取引回数: {summary['avg_total_trades']:.1f}回")
            print(f"利益期間数: {summary['profitable_periods']}回")
            print(f"損失期間数: {summary['losing_periods']}回")
    
    def save_results(self, filename: str = "walk_forward_results.csv"):
        """結果をCSVファイルに保存"""
        if not self.results:
            print("❌ 分析結果がありません。")
            return
        
        df_results = pd.DataFrame(self.results)
        
        # 日付列を文字列に変換（NaTやNoneは空文字に）
        for col in ['train_start', 'train_end', 'val_start', 'val_end']:
            df_results[col] = df_results[col].apply(safe_date_str)
        
        # パラメータ列を展開
        df_results['sma_short'] = df_results['best_params'].apply(lambda x: x['sma_short'])
        df_results['sma_medium'] = df_results['best_params'].apply(lambda x: x['sma_medium'])
        df_results['sma_long'] = df_results['best_params'].apply(lambda x: x['sma_long'])
        
        # best_params列を削除
        df_results = df_results.drop('best_params', axis=1)
        
        # CSVファイルに保存
        df_results.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✅ 結果を {filename} に保存しました。")


def main():
    """メイン実行関数"""
    print("=== Trading System Pro - ウォークフォワード分析 ===\n")
    
    # 設定
    stock_code = "7203.JP"  # トヨタ自動車
    initial_cash = 1_000_000  # 100万円
    
    # ウォークフォワード分析実行
    analyzer = WalkForwardAnalyzer(stock_code=stock_code, initial_cash=initial_cash)
    
    # 分析実行
    results = analyzer.run_walk_forward_analysis(
        training_period_years=2,
        validation_period_months=6,
        step_months=6,
        n_trials=30  # 時間短縮のため30回
    )
    
    if results:
        # 結果の可視化
        analyzer.plot_results()
        
        # 結果を保存
        analyzer.save_results()
        
        print("\n=== ウォークフォワード分析完了 ===")
    else:
        print("\n❌ ウォークフォワード分析に失敗しました。")


if __name__ == "__main__":
    main() 