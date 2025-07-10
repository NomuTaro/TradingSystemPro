# ==============================================================================
# --- Trading System Pro - 拡張機能クラス ---
# ==============================================================================

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt


class AdvancedPatternDetector:
    """酒田五法などの高度なパターン検出クラス"""
    
    def __init__(self):
        self.patterns_detected = []
    
    def detect_double_top(self, prices: pd.Series, threshold: float = 0.01) -> bool:
        """ダブルトップ（三山）を検出"""
        if len(prices) < 3:
            return False
        
        peaks = (prices.shift(1) < prices) & (prices.shift(-1) < prices)
        peak_indexes = prices[peaks].index
        if len(peak_indexes) < 2:
            return False
        
        p1, p2 = peak_indexes[-2], peak_indexes[-1]
        valley = prices[p1:p2].min()
        
        # 2つの山がほぼ同じ高さで、中間の谷が十分低いか
        if abs(prices[p1] - prices[p2]) / prices[p1] < threshold and valley < prices[p1] * (1 - threshold):
            return True
        return False

    def detect_double_bottom(self, prices: pd.Series, threshold: float = 0.01) -> bool:
        """ダブルボトム（三川）を検出"""
        if len(prices) < 3:
            return False
        
        troughs = (prices.shift(1) > prices) & (prices.shift(-1) > prices)
        trough_indexes = prices[troughs].index
        if len(trough_indexes) < 2:
            return False
        
        t1, t2 = trough_indexes[-2], trough_indexes[-1]
        peak = prices[t1:t2].max()
        
        # 2つの谷がほぼ同じ深さで、中間の山が十分高いか
        if abs(prices[t1] - prices[t2]) / prices[t1] < threshold and peak > prices[t1] * (1 + threshold):
            return True
        return False

    def detect_three_gap_up(self, df_window: pd.DataFrame) -> bool:
        """三空踏み上げを検出（直近5日間で判定）"""
        if len(df_window) < 4:
            return False
        
        gaps_up = 0
        for i in range(1, len(df_window)):
            # 窓を開けて上昇
            if df_window['Low'].iloc[i] > df_window['High'].iloc[i-1]:
                gaps_up += 1
        return gaps_up >= 3

    def detect_three_gap_down(self, df_window: pd.DataFrame) -> bool:
        """三空叩き込みを検出（直近5日間で判定）"""
        if len(df_window) < 4:
            return False
        
        gaps_down = 0
        for i in range(1, len(df_window)):
            # 窓を開けて下落
            if df_window['High'].iloc[i] < df_window['Low'].iloc[i-1]:
                gaps_down += 1
        return gaps_down >= 3

    def detect_three_white_soldiers(self, df_window: pd.DataFrame) -> bool:
        """赤三兵を検出（直近3日間）"""
        if len(df_window) != 3:
            return False
        
        # 3日連続で陽線
        is_all_positive = (df_window['Close'] > df_window['Open']).all()
        # 終値が日に日に上昇
        is_closing_up = (df_window['Close'].diff().dropna() > 0).all()
        # 始値が前日の実体の範囲内にある
        is_opening_in_body = (df_window['Open'].iloc[1] > df_window['Open'].iloc[0]) and \
                             (df_window['Open'].iloc[1] < df_window['Close'].iloc[0]) and \
                             (df_window['Open'].iloc[2] > df_window['Open'].iloc[1]) and \
                             (df_window['Open'].iloc[2] < df_window['Close'].iloc[1])
        return is_all_positive and is_closing_up and is_opening_in_body

    def detect_three_black_crows(self, df_window: pd.DataFrame) -> bool:
        """黒三兵（三羽烏）を検出（直近3日間）"""
        if len(df_window) != 3:
            return False
        
        # 3日連続で陰線
        is_all_negative = (df_window['Close'] < df_window['Open']).all()
        # 終値が日に日に下落
        is_closing_down = (df_window['Close'].diff().dropna() < 0).all()
        # 始値が前日の実体の範囲内にある
        is_opening_in_body = (df_window['Open'].iloc[1] < df_window['Open'].iloc[0]) and \
                             (df_window['Open'].iloc[1] > df_window['Close'].iloc[0]) and \
                             (df_window['Open'].iloc[2] < df_window['Open'].iloc[1]) and \
                             (df_window['Open'].iloc[2] > df_window['Close'].iloc[1])
        return is_all_negative and is_closing_down and is_opening_in_body


class PerformanceAnalyzer:
    """パフォーマンス分析クラス"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_performance_metrics(self, asset_history: List[float], initial_cash: float) -> Dict[str, Any]:
        """パフォーマンス指標を計算"""
        if not asset_history or len(asset_history) < 2:
            return {}
        
        final_value = asset_history[-1]
        total_return = (final_value - initial_cash) / initial_cash
        
        # 日次リターンの計算
        daily_returns = []
        for i in range(1, len(asset_history)):
            daily_return = (asset_history[i] - asset_history[i-1]) / asset_history[i-1]
            daily_returns.append(daily_return)
        
        if daily_returns:
            mean_return = np.mean(daily_returns)
            volatility = np.std(daily_returns)
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0
            
            # 最大ドローダウン
            peak = asset_history[0]
            max_drawdown = 0
            for value in asset_history:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        else:
            mean_return = volatility = sharpe_ratio = max_drawdown = 0
        
        return {
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (252 / len(asset_history)) - 1,
            'volatility': volatility * np.sqrt(252),  # 年率換算
            'sharpe_ratio': sharpe_ratio * np.sqrt(252),  # 年率換算
            'max_drawdown': max_drawdown,
            'final_value': final_value,
            'profit_loss': final_value - initial_cash
        }
    
    def analyze_trades(self, trade_history: List[str]) -> Dict[str, Any]:
        """取引履歴を分析"""
        if not trade_history:
            return {}
        
        total_trades = len(trade_history)
        profitable_trades = sum(1 for trade in trade_history if 'profit' in trade.lower())
        
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'losing_trades': total_trades - profitable_trades,
            'win_rate': win_rate,
            'average_trades_per_period': total_trades / 252 if total_trades > 0 else 0
        }
    
    def calculate_risk_metrics(self, daily_returns: List[float]) -> Dict[str, Any]:
        """リスク指標を計算"""
        if not daily_returns:
            return {}
        
        returns_array = np.array(daily_returns)
        
        # VaR (Value at Risk) - 5%信頼水準
        var_95 = np.percentile(returns_array, 5)
        
        # CVaR (Conditional Value at Risk)
        cvar_95 = returns_array[returns_array <= var_95].mean()
        
        return {
            'value_at_risk_95': var_95,
            'conditional_var_95': cvar_95,
            'downside_deviation': np.std(returns_array[returns_array < 0]),
            'skewness': float(pd.Series(returns_array).skew()),
            'kurtosis': float(pd.Series(returns_array).kurtosis())
        }


class RiskManager:
    """リスク管理クラス"""
    
    def __init__(self, max_position_size: float = 0.1, max_portfolio_risk: float = 0.02):
        self.max_position_size = max_position_size  # ポートフォリオの最大10%
        self.max_portfolio_risk = max_portfolio_risk  # ポートフォリオの最大2%リスク
    
    def calculate_position_size(self, portfolio_value: float, entry_price: float, stop_loss_distance: float) -> int:
        """ポジションサイズを計算"""
        # リスクベースのポジションサイズ
        risk_amount = portfolio_value * self.max_portfolio_risk
        position_size_risk = risk_amount / stop_loss_distance
        
        # 最大ポジションサイズ制限
        max_position_value = portfolio_value * self.max_position_size
        position_size_max = max_position_value / entry_price
        
        # 小さい方を選択
        position_size = min(position_size_risk, position_size_max)
        
        return int(max(0, position_size))
    
    def calculate_stop_loss(self, entry_price: float, atr: float, position_type: str, multiplier: float = 2.0) -> float:
        """ストップロス価格を計算"""
        if position_type.lower() == 'long':
            return entry_price - (atr * multiplier)
        else:  # short
            return entry_price + (atr * multiplier)
    
    def calculate_take_profit(self, entry_price: float, atr: float, position_type: str, multiplier: float = 3.0) -> float:
        """テイクプロフィット価格を計算"""
        if position_type.lower() == 'long':
            return entry_price + (atr * multiplier)
        else:  # short
            return entry_price - (atr * multiplier)
    
    def calculate_max_drawdown(self, asset_history: List[float]) -> float:
        """最大ドローダウンを計算"""
        if len(asset_history) < 2:
            return 0.0
        
        peak = asset_history[0]
        max_drawdown = 0.0
        
        for value in asset_history:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return max_drawdown
    
    def calculate_current_drawdown(self, asset_history: List[float]) -> float:
        """現在のドローダウンを計算"""
        if len(asset_history) < 2:
            return 0.0
        
        peak = max(asset_history)
        current_value = asset_history[-1]
        
        return (peak - current_value) / peak
    
    def get_risk_recommendations(self, portfolio_value: float, current_drawdown: float, trade_count: int) -> List[str]:
        """リスク管理の推奨事項を提供"""
        recommendations = []
        
        if current_drawdown > 0.15:
            recommendations.append("⚠️ 重要: ドローダウンが15%を超えています。ポジションサイズを縮小してください。")
        elif current_drawdown > 0.10:
            recommendations.append("⚠️ 警告: ドローダウンが10%を超えています。リスク管理を見直してください。")
        
        if trade_count > 100:
            recommendations.append("📈 十分な取引履歴があります。パフォーマンス分析を詳しく行ってください。")
        elif trade_count < 20:
            recommendations.append("📊 取引履歴が少ないです。より多くのデータを蓄積してから判断してください。")
        
        if portfolio_value < 10000:
            recommendations.append("💰 資金が少額です。リスクを最小限に抑えた取引を心がけてください。")
        
        if not recommendations:
            recommendations.append("✅ 現在のリスク状況は良好です。")
        
        return recommendations


class Optimizer:
    """パラメータ最適化クラス"""
    
    def __init__(self, trading_system_class, stock_code: str = "AAPL"):
        """
        Optimizerの初期化
        
        Args:
            trading_system_class: TradingSystemクラス（インスタンスではなくクラス自体）
            stock_code: 最適化に使用する銘柄コード
        """
        self.trading_system_class = trading_system_class
        self.stock_code = stock_code
        self.results = []
    
    def grid_search(self, param_ranges: Dict[str, Any], 
                   objective: str = "final_cash", 
                   minimize: bool = False) -> pd.DataFrame:
        """
        グリッドサーチによるパラメータ最適化
        
        Args:
            param_ranges: 最適化するパラメータの範囲
                例: {
                    'BUY_THRESHOLD': [1.0, 1.5, 2.0, 2.5, 3.0],
                    'SELL_THRESHOLD': [1.0, 1.5, 2.0, 2.5, 3.0],
                    'SIGNAL_WEIGHTS': {
                        'golden_cross_short': [1.0, 1.5, 2.0]
                    }
                }
            objective: 最適化の目標指標 ('final_cash', 'sharpe_ratio', 'max_drawdown')
            minimize: Trueの場合は目標指標を最小化、Falseの場合は最大化
        
        Returns:
            pd.DataFrame: 全試行結果をまとめたDataFrame
        """
        import itertools
        from copy import deepcopy
        
        print(f"=== グリッドサーチ開始 ===")
        print(f"銘柄: {self.stock_code}")
        print(f"最適化目標: {objective} ({'minimize' if minimize else 'maximize'})")
        
        # パラメータの組み合わせを生成
        param_combinations = self._generate_param_combinations(param_ranges)
        total_combinations = len(param_combinations)
        
        print(f"試行回数: {total_combinations}")
        print("最適化実行中...")
        
        self.results = []
        
        for i, params in enumerate(param_combinations):
            try:
                # TradingSystemインスタンスを作成
                system = self.trading_system_class(self.stock_code)
                
                # パラメータを設定
                self._apply_parameters(system, params)
                
                # バックテスト実行
                system.prepare_data()
                system.backtest()
                
                # 評価指標を計算
                metrics = self._calculate_metrics(system)
                
                # 結果を記録
                result = {**params, **metrics}
                self.results.append(result)
                
                # 進捗表示
                if (i + 1) % max(1, total_combinations // 10) == 0:
                    progress = (i + 1) / total_combinations * 100
                    print(f"進捗: {progress:.1f}% ({i + 1}/{total_combinations})")
                    
            except Exception as e:
                print(f"パラメータ組み合わせ {i+1} でエラー: {e}")
                continue
        
        # 結果をDataFrameに変換
        results_df = pd.DataFrame(self.results)
        
        if len(results_df) == 0:
            print("❌ 有効な結果が得られませんでした")
            return pd.DataFrame()
        
        # 最適なパラメータを特定
        if minimize:
            best_idx = results_df[objective].idxmin()
        else:
            best_idx = results_df[objective].idxmax()
        
        best_params = results_df.loc[best_idx]
        
        print(f"\n=== 最適化結果 ===")
        print(f"最適な{objective}: {best_params[objective]:.4f}")
        print(f"最適なパラメータ:")
        
        # パラメータのみを表示
        param_keys = list(param_ranges.keys())
        for key in param_keys:
            if key in best_params:
                print(f"  {key}: {best_params[key]}")
        
        # 結果を性能順にソート
        results_df_sorted = results_df.sort_values(
            objective, ascending=minimize
        ).reset_index(drop=True)
        
        return results_df_sorted
    
    def _generate_param_combinations(self, param_ranges: Dict[str, Any]) -> List[Dict]:
        """パラメータの全組み合わせを生成"""
        import itertools
        
        # フラットなパラメータと辞書パラメータを分離
        flat_params = {}
        dict_params = {}
        
        for key, value in param_ranges.items():
            if isinstance(value, dict):
                dict_params[key] = value
            else:
                flat_params[key] = value
        
        combinations = []
        
        # フラットなパラメータの組み合わせを生成
        if flat_params:
            keys = list(flat_params.keys())
            values = list(flat_params.values())
            
            for combination in itertools.product(*values):
                param_dict = dict(zip(keys, combination))
                
                # 辞書パラメータの組み合わせを追加
                if dict_params:
                    dict_combinations = self._generate_dict_combinations(dict_params)
                    for dict_combo in dict_combinations:
                        full_param = {**param_dict, **dict_combo}
                        combinations.append(full_param)
                else:
                    combinations.append(param_dict)
        else:
            # フラットなパラメータがない場合は辞書パラメータのみ
            combinations = self._generate_dict_combinations(dict_params)
        
        return combinations
    
    def _generate_dict_combinations(self, dict_params: Dict[str, Dict]) -> List[Dict]:
        """辞書型パラメータの組み合わせを生成"""
        import itertools
        
        combinations = []
        
        for main_key, sub_params in dict_params.items():
            sub_keys = list(sub_params.keys())
            sub_values = list(sub_params.values())
            
            for combination in itertools.product(*sub_values):
                param_dict = {main_key: dict(zip(sub_keys, combination))}
                combinations.append(param_dict)
        
        return combinations
    
    def _apply_parameters(self, system, params: Dict):
        """TradingSystemインスタンスにパラメータを適用"""
        for key, value in params.items():
            if hasattr(system, key.lower()):
                setattr(system, key.lower(), value)
            elif hasattr(system, key):
                setattr(system, key, value)
            elif key == 'SIGNAL_WEIGHTS' and hasattr(system, 'signal_weights'):
                # SIGNAL_WEIGHTSの場合は既存の辞書を更新
                system.signal_weights.update(value)
    
    def _calculate_metrics(self, system) -> Dict[str, float]:
        """評価指標を計算"""
        metrics = {}
        
        # 基本指標
        metrics['final_cash'] = system.final_cash
        metrics['total_return'] = (system.final_cash - system.initial_cash) / system.initial_cash
        metrics['trade_count'] = len(system.trade_history)
        
        # 資産履歴を取得
        if system.asset_history:
            if isinstance(system.asset_history[0], tuple):
                asset_values = [asset[1] for asset in system.asset_history]
            else:
                asset_values = system.asset_history
            
            # 日次リターンを計算
            daily_returns = []
            for i in range(1, len(asset_values)):
                daily_return = (asset_values[i] - asset_values[i-1]) / asset_values[i-1]
                daily_returns.append(daily_return)
            
            if daily_returns:
                # シャープレシオ
                mean_return = np.mean(daily_returns)
                std_return = np.std(daily_returns)
                metrics['sharpe_ratio'] = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0
                
                # 最大ドローダウン
                peak = asset_values[0]
                max_drawdown = 0
                for value in asset_values:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                
                metrics['max_drawdown'] = max_drawdown
                metrics['volatility'] = std_return * np.sqrt(252)
                
                # 勝率
                profitable_trades = 0
                for trade in system.trade_history:
                    if 'type' in trade and trade['type'] == 'SELL':
                        # 簡易的な損益判定（実際の実装に応じて調整）
                        if 'reason' in trade and 'profit' not in trade['reason'].lower():
                            profitable_trades += 1
                
                metrics['win_rate'] = profitable_trades / len(system.trade_history) if system.trade_history else 0
            else:
                metrics['sharpe_ratio'] = 0
                metrics['max_drawdown'] = 0
                metrics['volatility'] = 0
                metrics['win_rate'] = 0
        else:
            metrics['sharpe_ratio'] = 0
            metrics['max_drawdown'] = 0
            metrics['volatility'] = 0
            metrics['win_rate'] = 0
        
        return metrics
    
    def get_best_parameters(self, objective: str = "final_cash", minimize: bool = False) -> Dict:
        """最適なパラメータを取得"""
        if not self.results:
            raise ValueError("最適化が実行されていません。先にgrid_search()を実行してください。")
        
        results_df = pd.DataFrame(self.results)
        
        if minimize:
            best_idx = results_df[objective].idxmin()
        else:
            best_idx = results_df[objective].idxmax()
        
        return results_df.loc[best_idx].to_dict()
    
    def plot_optimization_results(self, x_param: str, y_param: str, 
                                 objective: str = "final_cash") -> None:
        """最適化結果を可視化"""
        if not self.results:
            raise ValueError("最適化が実行されていません。")
        
        results_df = pd.DataFrame(self.results)
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(results_df[x_param], results_df[y_param], 
                            c=results_df[objective], cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label=objective)
        plt.xlabel(x_param)
        plt.ylabel(y_param)
        plt.title(f'最適化結果: {objective}')
        plt.grid(True, alpha=0.3)
        plt.show()
