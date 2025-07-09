# ==============================================================================
# --- Trading System Pro - メインクラス ---
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas_ta as ta
from pandas_datareader import data
import warnings
from typing import List, Dict, Tuple, Optional, Any
from . import config

warnings.simplefilter('ignore')

# ==============================================================================
# --- 酒田五法の判定関数 ---
# ==============================================================================

def detect_double_top(prices, threshold=0.01):
    """ダブルトップ（三山）を検出"""
    if len(prices) < 3: return False
    peaks = (prices.shift(1) < prices) & (prices.shift(-1) < prices)
    peak_indexes = prices[peaks].index
    if len(peak_indexes) < 2: return False
    
    p1, p2 = peak_indexes[-2], peak_indexes[-1]
    valley = prices[p1:p2].min()
    
    # 2つの山がほぼ同じ高さで、中間の谷が十分低いか
    if abs(prices[p1] - prices[p2]) / prices[p1] < threshold and valley < prices[p1] * (1 - threshold):
        return True
    return False

def detect_double_bottom(prices, threshold=0.01):
    """ダブルボトム（三川）を検出"""
    if len(prices) < 3: return False
    troughs = (prices.shift(1) > prices) & (prices.shift(-1) > prices)
    trough_indexes = prices[troughs].index
    if len(trough_indexes) < 2: return False
    
    t1, t2 = trough_indexes[-2], trough_indexes[-1]
    peak = prices[t1:t2].max()
    
    # 2つの谷がほぼ同じ深さで、中間の山が十分高いか
    if abs(prices[t1] - prices[t2]) / prices[t1] < threshold and peak > prices[t1] * (1 + threshold):
        return True
    return False

def detect_three_gap_up(df_window):
    """三空踏み上げを検出（直近5日間で判定）"""
    if len(df_window) < 4: return False
    gaps_up = 0
    for i in range(1, len(df_window)):
        # 窓を開けて上昇
        if df_window['Low'].iloc[i] > df_window['High'].iloc[i-1]:
            gaps_up += 1
    return gaps_up >= 3

def detect_three_gap_down(df_window):
    """三空叩き込みを検出（直近5日間で判定）"""
    if len(df_window) < 4: return False
    gaps_down = 0
    for i in range(1, len(df_window)):
        # 窓を開けて下落
        if df_window['High'].iloc[i] < df_window['Low'].iloc[i-1]:
            gaps_down += 1
    return gaps_down >= 3

def detect_three_white_soldiers(df_window):
    """赤三兵を検出（直近3日間）"""
    if len(df_window) != 3: return False
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

def detect_three_black_crows(df_window):
    """黒三兵（三羽烏）を検出（直近3日間）"""
    if len(df_window) != 3: return False
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


# ==============================================================================
# --- TradingSystemクラス ---
# ==============================================================================

class TradingSystem:
    """株式取引システムのメインクラス"""
    
    def __init__(self, stock_code: Optional[str] = None):
        """
        TradingSystemの初期化
        
        Args:
            stock_code (str, optional): 銘柄コード。Noneの場合はconfig.pyの設定を使用
        """
        # 設定項目の読み込み
        self.stock_code = stock_code or config.STOCK_CODE
        self.data_period_days = config.DATA_PERIOD_DAYS
        self.initial_cash = config.INITIAL_CASH
        self.investment_ratio = config.INVESTMENT_RATIO
        self.fee_rate = config.FEE_RATE
        self.slippage_rate = config.SLIPPAGE_RATE
        self.take_profit_atr_multiple = config.TAKE_PROFIT_ATR_MULTIPLE
        self.stop_loss_atr_multiple = config.STOP_LOSS_ATR_MULTIPLE
        self.take_profit_rate = config.TAKE_PROFIT_RATE
        self.stop_loss_rate = config.STOP_LOSS_RATE
        self.buy_threshold = config.BUY_THRESHOLD
        self.sell_threshold = config.SELL_THRESHOLD
        self.signal_weights = config.SIGNAL_WEIGHTS
        
        # 状態変数の初期化
        self.df: Optional[pd.DataFrame] = None
        self.asset_history: List[Tuple] = []
        self.trade_history: List[Dict] = []
        self.final_cash: float = 0.0
    
    def prepare_data(self) -> Optional[pd.DataFrame]:
        """株価データを取得し、テクニカル指標を計算する"""
        print(f"Loading data for {self.stock_code}...")
        try:
            df = data.DataReader(self.stock_code, 'stooq')
            if df.empty:
                print(f"エラー: {self.stock_code} のデータが見つかりません。")
                return None
        except Exception as e:
            print(f"データ取得エラー: {e}")
            return None
            
        df = df.sort_index()

        # pandas_taを使用してテクニカル指標を計算
        df.ta.sma(length=5, append=True)
        df.ta.sma(length=25, append=True)
        df.ta.sma(length=75, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.bbands(length=25, std=2, append=True)
        df.ta.atr(length=14, append=True)
        
        # 不要なカラムを削除する前に統一カラムの存在確認
        print("テクニカル指標計算後の列:", [col for col in df.columns if any(x in col for x in ['SMA', 'RSI', 'MACD', 'BB', 'ATR'])])
        
        # カラム名を統一（実際に存在する列のみ）
        if 'SMA_5' in df.columns:
            df['sma5'] = df['SMA_5']
        if 'SMA_25' in df.columns:
            df['sma25'] = df['SMA_25']
        if 'SMA_75' in df.columns:
            df['sma75'] = df['SMA_75']
        else:
            # SMA_75が計算されていない場合、手動で計算
            df['sma75'] = df['Close'].rolling(window=75).mean()
        if 'RSI_14' in df.columns:
            df['RSI'] = df['RSI_14']
        if 'MACD_12_26_9' in df.columns:
            df['MACD'] = df['MACD_12_26_9']
        if 'MACDs_12_26_9' in df.columns:
            df['MACD_signal'] = df['MACDs_12_26_9']
        if 'MACDh_12_26_9' in df.columns:
            df['MACD_hist'] = df['MACDh_12_26_9']
        if 'BBU_25_2.0' in df.columns:
            df['BB_upper'] = df['BBU_25_2.0']
        if 'BBM_25_2.0' in df.columns:
            df['BB_middle'] = df['BBM_25_2.0']
        if 'BBL_25_2.0' in df.columns:
            df['BB_lower'] = df['BBL_25_2.0']
        if 'ATRr_14' in df.columns:
            df['ATR'] = df['ATRr_14']
        
        # 必要な列が存在するかチェック
        required_cols = ['sma5', 'sma25', 'sma75', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 
                        'BB_upper', 'BB_middle', 'BB_lower', 'ATR']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"警告: 以下の列が見つかりません: {missing_cols}")
        
        # 不要なカラムを削除
        cols_to_drop = [col for col in df.columns if col.startswith(('SMA_', 'RSI_', 'MACD', 'BB', 'ATR')) 
                        and col not in ['RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'ATR']]
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        
        # データを指定期間に絞り、欠損値がある行を削除
        df = df.tail(self.data_period_days).dropna()
        self.df = df
        return df
    
    def evaluate_buy_signals(self, df_hist: pd.DataFrame, current_idx: int) -> float:
        """
        過去データのみを使用して買いシグナルを評価
        df_hist: current_idxまでの過去データ
        current_idx: 現在のインデックス（実際の取引実行日）
        """
        buy_signal_score = 0.0
        
        if len(df_hist) < 75:  # 最低限必要なデータ数
            return buy_signal_score
        
        # 前日（i-1）時点のデータを使用
        prev_day = df_hist.iloc[-1]
        prev_day_2 = df_hist.iloc[-2] if len(df_hist) >= 2 else None
        
        # テクニカル指標シグナル
        if prev_day_2 is not None:
            # ゴールデンクロス（短期）
            if ('sma5' in df_hist.columns and 'sma25' in df_hist.columns and
                prev_day_2['sma5'] < prev_day_2['sma25'] and 
                prev_day['sma5'] > prev_day['sma25']):
                buy_signal_score += self.signal_weights['golden_cross_short']
            
            # ゴールデンクロス（長期）
            if ('sma25' in df_hist.columns and 'sma75' in df_hist.columns and
                prev_day_2['sma25'] < prev_day_2['sma75'] and 
                prev_day['sma25'] > prev_day['sma75']):
                buy_signal_score += self.signal_weights['golden_cross_long']
            
            # MACDブリッシュクロス
            if ('MACD' in df_hist.columns and 'MACD_signal' in df_hist.columns and
                prev_day_2['MACD'] < prev_day_2['MACD_signal'] and 
                prev_day['MACD'] > prev_day['MACD_signal']):
                buy_signal_score += self.signal_weights['macd_bullish']
        
        # ボリンジャーバンド下限突破
        if 'BB_lower' in df_hist.columns and prev_day['Close'] < prev_day['BB_lower']:
            buy_signal_score += self.signal_weights['bb_oversold']
        
        # RSI売られすぎ
        if 'RSI' in df_hist.columns and prev_day['RSI'] < 30:
            buy_signal_score += self.signal_weights['rsi_oversold']
        
        # 酒田五法シグナル（過去データのみ使用）
        if len(df_hist) >= 20:
            prices_hist = df_hist['Close'].iloc[-20:]  # 過去20日分
            if detect_double_bottom(prices_hist):
                buy_signal_score += self.signal_weights['double_bottom']
        
        if len(df_hist) >= 3:
            df_3days = df_hist.iloc[-3:]
            if detect_three_white_soldiers(df_3days):
                buy_signal_score += self.signal_weights['three_white_soldiers']
        
        if len(df_hist) >= 5:
            df_5days = df_hist.iloc[-5:]
            if detect_three_gap_down(df_5days):
                buy_signal_score += self.signal_weights['three_gap_down']
        
        return buy_signal_score

    def evaluate_sell_signals(self, df_hist: pd.DataFrame, current_idx: int) -> float:
        """
        過去データのみを使用して売りシグナルを評価
        df_hist: current_idxまでの過去データ
        current_idx: 現在のインデックス（実際の取引実行日）
        """
        sell_signal_score = 0.0
        
        if len(df_hist) < 75:  # 最低限必要なデータ数
            return sell_signal_score
        
        # 前日（i-1）時点のデータを使用
        prev_day = df_hist.iloc[-1]
        prev_day_2 = df_hist.iloc[-2] if len(df_hist) >= 2 else None
        
        # テクニカル指標シグナル
        if prev_day_2 is not None:
            # デッドクロス（短期）
            if ('sma5' in df_hist.columns and 'sma25' in df_hist.columns and
                prev_day_2['sma5'] > prev_day_2['sma25'] and 
                prev_day['sma5'] < prev_day['sma25']):
                sell_signal_score += self.signal_weights['dead_cross_short']
            
            # デッドクロス（長期）
            if ('sma25' in df_hist.columns and 'sma75' in df_hist.columns and
                prev_day_2['sma25'] > prev_day_2['sma75'] and 
                prev_day['sma25'] < prev_day['sma75']):
                sell_signal_score += self.signal_weights['dead_cross_long']
            
            # MACDベアリッシュクロス
            if ('MACD' in df_hist.columns and 'MACD_signal' in df_hist.columns and
                prev_day_2['MACD'] > prev_day_2['MACD_signal'] and 
                prev_day['MACD'] < prev_day['MACD_signal']):
                sell_signal_score += self.signal_weights['macd_bearish']
        
        # ボリンジャーバンド上限突破
        if 'BB_upper' in df_hist.columns and prev_day['Close'] > prev_day['BB_upper']:
            sell_signal_score += self.signal_weights['bb_overbought']
        
        # RSI買われすぎ
        if 'RSI' in df_hist.columns and prev_day['RSI'] > 70:
            sell_signal_score += self.signal_weights['rsi_overbought']
        
        # 酒田五法シグナル（過去データのみ使用）
        if len(df_hist) >= 20:
            prices_hist = df_hist['Close'].iloc[-20:]  # 過去20日分
            if detect_double_top(prices_hist):
                sell_signal_score += self.signal_weights['double_top']
        
        if len(df_hist) >= 3:
            df_3days = df_hist.iloc[-3:]
            if detect_three_black_crows(df_3days):
                sell_signal_score += self.signal_weights['three_black_crows']
        
        if len(df_hist) >= 5:
            df_5days = df_hist.iloc[-5:]
            if detect_three_gap_up(df_5days):
                sell_signal_score += self.signal_weights['three_gap_up']
        
        return sell_signal_score

    def run_simulation(self) -> Tuple[Optional[List[Tuple]], Optional[List[Dict]], float]:
        """シグナルに基づき売買シミュレーションを実行する（ルックアヘッドバイアス排除版）"""
        if self.df is None:
            print("エラー: データが準備されていません。prepare_data()を先に実行してください。")
            return None, None, 0.0
        
        cash = self.initial_cash
        position = 0
        buy_price = 0.0
        buy_atr = 0.0  # 購入時のATR
        in_position = False
        asset_history = []
        trade_history = []

        print("Running simulation...")
        
        # 最低限必要なデータ数を確保
        start_idx = max(75, len(self.df) // 4)  # 全データの1/4以降から開始
        
        for i in range(start_idx, len(self.df)):
            today = self.df.index[i]
            today_data = self.df.iloc[i]
            
            # 過去データのみを使用（i日目のデータは含まない）
            df_hist = self.df.iloc[0:i]
            
            # 売買執行価格（スリッページ考慮）
            execution_price_buy = today_data['Close'] * (1 + self.slippage_rate)
            execution_price_sell = today_data['Close'] * (1 - self.slippage_rate)

            # --- シグナル評価（過去データのみ使用） ---
            buy_signal_score = self.evaluate_buy_signals(df_hist, i)
            sell_signal_score = self.evaluate_sell_signals(df_hist, i)

            # --- 売買判断 ---
            if not in_position:
                # 買い判断
                if buy_signal_score >= self.buy_threshold:
                    invest_amount = cash * self.investment_ratio
                    qty_to_buy = int(invest_amount / (execution_price_buy * (1 + self.fee_rate)))
                    
                    if qty_to_buy * execution_price_buy * (1 + self.fee_rate) <= cash and qty_to_buy > 0:
                        position = qty_to_buy
                        cash -= position * execution_price_buy * (1 + self.fee_rate)
                        buy_price = execution_price_buy
                        buy_atr = df_hist.iloc[-1]['ATR'] if len(df_hist) > 0 else 0  # 購入時のATR
                        in_position = True
                        trade_history.append({
                            'type': 'BUY', 
                            'date': today, 
                            'price': execution_price_buy, 
                            'qty': position,
                            'signal_score': buy_signal_score
                        })
            else:
                # 売り判断（シグナル or 利食い or 損切り）
                sell_reason = None
                
                # シグナルによる売り判断
                if sell_signal_score >= self.sell_threshold:
                    sell_reason = f'Signal (score: {sell_signal_score:.1f})'
                else:
                    # ATRベースの利食い・損切り判定
                    if self.take_profit_atr_multiple > 0 and buy_atr > 0:
                        take_profit_price = buy_price + (buy_atr * self.take_profit_atr_multiple)
                        if today_data['Close'] >= take_profit_price:
                            sell_reason = f'Take Profit (ATR x{self.take_profit_atr_multiple})'
                    
                    if self.stop_loss_atr_multiple > 0 and buy_atr > 0:
                        stop_loss_price = buy_price - (buy_atr * self.stop_loss_atr_multiple)
                        if today_data['Close'] <= stop_loss_price:
                            sell_reason = f'Stop Loss (ATR x{self.stop_loss_atr_multiple})'
                    
                    # ATR設定が無効な場合、固定率を使用
                    if not sell_reason:
                        if self.take_profit_rate > 0:
                            take_profit_price = buy_price * (1 + self.take_profit_rate)
                            if today_data['Close'] >= take_profit_price:
                                sell_reason = f'Take Profit ({self.take_profit_rate*100:.1f}%)'
                        
                        if self.stop_loss_rate > 0:
                            stop_loss_price = buy_price * (1 - self.stop_loss_rate)
                            if today_data['Close'] <= stop_loss_price:
                                sell_reason = f'Stop Loss ({self.stop_loss_rate*100:.1f}%)'

                if sell_reason:
                    cash += position * execution_price_sell * (1 - self.fee_rate)
                    trade_history.append({
                        'type': 'SELL', 
                        'date': today, 
                        'price': execution_price_sell, 
                        'qty': position, 
                        'reason': sell_reason
                    })
                    position = 0
                    buy_price = 0.0
                    buy_atr = 0.0
                    in_position = False

            # 資産評価
            current_asset = cash + (position * today_data['Close'] if in_position else 0)
            asset_history.append((today, current_asset))

        # 最終日にポジションがあれば強制決済
        if in_position:
            final_price = self.df['Close'].iloc[-1] * (1 - self.slippage_rate)  # スリッページ考慮
            cash += position * final_price * (1 - self.fee_rate)
            trade_history.append({
                'type': 'SELL', 
                'date': self.df.index[-1], 
                'price': final_price, 
                'qty': position, 
                'reason': 'End of Simulation'
            })
            asset_history[-1] = (self.df.index[-1], cash)
        
        # インスタンス変数に保存
        self.asset_history = asset_history
        self.trade_history = trade_history
        self.final_cash = cash
        
        return asset_history, trade_history, cash

    def show_results(self) -> None:
        """シミュレーション結果をチャートとテキストで表示する"""
        if self.df is None or not self.asset_history:
            print("エラー: シミュレーションが実行されていません。")
            return
        
        print("\n" + "="*50)
        print("📈 シミュレーション結果")
        print("="*50)

        # --- 取引履歴と損益サマリー ---
        positions = []
        buy_info = {}
        total_trades = 0
        winning_trades = 0
        total_profit = 0
        total_loss = 0

        for trade in self.trade_history:
            if trade['type'] == 'BUY':
                buy_info = trade
            elif trade['type'] == 'SELL':
                if buy_info:
                    buy_cost = buy_info['qty'] * buy_info['price']
                    sell_revenue = trade['qty'] * trade['price']
                    profit = (sell_revenue * (1 - self.fee_rate)) - (buy_cost * (1 + self.fee_rate))
                    
                    positions.append({
                        'buy_date': buy_info['date'], 'buy_price': buy_info['price'],
                        'sell_date': trade['date'], 'sell_price': trade['price'],
                        'qty': trade['qty'], 'profit': profit, 'reason': trade.get('reason', 'N/A'),
                        'buy_signal_score': buy_info.get('signal_score', 0)
                    })
                    
                    total_trades += 1
                    if profit > 0:
                        winning_trades += 1
                        total_profit += profit
                    else:
                        total_loss += profit
                    buy_info = {} # Reset for next trade

        print("\n📊【取引履歴】")
        if not positions:
            print("取引は発生しませんでした。")
        else:
            for p in positions:
                result_icon = "✅" if p['profit'] > 0 else "❌"
                print(f"{result_icon} [{p['reason']}] "
                      f"{p['buy_date'].strftime('%y-%m-%d')} 買: {p['buy_price']:,.0f} (score: {p['buy_signal_score']:.1f}) → "
                      f"{p['sell_date'].strftime('%y-%m-%d')} 売: {p['sell_price']:,.0f} | "
                      f"数量: {p['qty']:,.0f} | 損益: {p['profit']:,.0f}円")

        # --- パフォーマンスサマリー ---
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        avg_profit_loss = (total_profit + total_loss) / total_trades if total_trades > 0 else 0
        
        print("\n📝【パフォーマンスサマリー】")
        print(f"💰 最終資産: {self.final_cash:,.0f}円 (初期資金: {self.initial_cash:,.0f}円)")
        print(f"📈 総損益: {self.final_cash - self.initial_cash:,.0f}円")
        print(f"🔄 トレード数: {total_trades} 回")
        print(f"🎯 勝率: {win_rate:.2f}% ({winning_trades}勝 / {total_trades - winning_trades}敗)")
        print(f"⚖️ プロフィットファクター: {profit_factor:.2f}")
        print(f"💹 平均損益: {avg_profit_loss:,.0f}円")

        # --- Buy & Hold戦略との比較 ---
        bh_qty = int((self.initial_cash * self.investment_ratio) / (self.df['Close'].iloc[0] * (1 + self.fee_rate)))
        if bh_qty > 0:
            bh_buy_cost = bh_qty * self.df['Close'].iloc[0] * (1 + self.fee_rate)
            bh_sell_revenue = bh_qty * self.df['Close'].iloc[-1] * (1 - self.fee_rate)
            bh_profit = bh_sell_revenue - bh_buy_cost
            bh_remaining_cash = self.initial_cash - bh_buy_cost
            bh_total = bh_remaining_cash + bh_sell_revenue
            
            print("\n⚖️【Buy & Hold戦略との比較】")
            print(f"アルゴリズム戦略の総損益: {self.final_cash - self.initial_cash:,.0f}円")
            print(f"Buy & Hold戦略の総損益 : {bh_total - self.initial_cash:,.0f}円")
            if (self.final_cash - self.initial_cash) > (bh_total - self.initial_cash):
                print("✅ あなたの戦略は Buy & Hold より優れています。")
            else:
                print("❌ Buy & Hold の方が良い結果でした。")

        # --- チャート描画 ---
        self._plot_charts()

        # --- 資産推移の描画 ---
        self._plot_asset_history()
    
    def _plot_charts(self) -> None:
        """メインチャートの描画（内部メソッド）"""
        # 基本のプロット要素
        addplots = []
        
        # 移動平均線
        if 'sma5' in self.df.columns:
            addplots.append(mpf.make_addplot(self.df['sma5'], color='green', width=0.7))
        if 'sma25' in self.df.columns:
            addplots.append(mpf.make_addplot(self.df['sma25'], color='orange', width=0.7))
        if 'sma75' in self.df.columns:
            addplots.append(mpf.make_addplot(self.df['sma75'], color='purple', width=0.7))
        
        # ボリンジャーバンド
        if 'BB_upper' in self.df.columns:
            addplots.append(mpf.make_addplot(self.df['BB_upper'], color='grey', linestyle='--', width=0.5))
        if 'BB_lower' in self.df.columns:
            addplots.append(mpf.make_addplot(self.df['BB_lower'], color='grey', linestyle='--', width=0.5))
        
        # MACD (パネル2)
        if 'MACD_hist' in self.df.columns:
            addplots.append(mpf.make_addplot(self.df['MACD_hist'], type='bar', panel=1, color='gray', alpha=0.5))
        if 'MACD' in self.df.columns:
            addplots.append(mpf.make_addplot(self.df['MACD'], panel=1, color='blue', width=0.7))
        if 'MACD_signal' in self.df.columns:
            addplots.append(mpf.make_addplot(self.df['MACD_signal'], panel=1, color='red', width=0.7))
        
        # RSI (パネル3)
        if 'RSI' in self.df.columns:
            addplots.append(mpf.make_addplot(self.df['RSI'], panel=2, color='purple', width=0.7, ylim=(0, 100)))
            addplots.append(mpf.make_addplot([70]*len(self.df), panel=2, color='red', linestyle='--', width=0.5))
            addplots.append(mpf.make_addplot([30]*len(self.df), panel=2, color='green', linestyle='--', width=0.5))
        
        # 売買マーカーを追加
        buy_markers = []
        sell_markers = []
        
        for trade in self.trade_history:
            if trade['type'] == 'BUY':
                try:
                    idx = self.df.index.get_loc(trade['date'])
                    buy_markers.append(idx)
                except KeyError:
                    continue
            elif trade['type'] == 'SELL':
                try:
                    idx = self.df.index.get_loc(trade['date'])
                    sell_markers.append(idx)
                except KeyError:
                    continue
        
        # 買いマーカー
        if buy_markers:
            buy_prices = [self.df['Low'].iloc[i] * 0.98 for i in buy_markers]
            buy_plot = [np.nan] * len(self.df)
            for i, price in zip(buy_markers, buy_prices):
                buy_plot[i] = price
            addplots.append(mpf.make_addplot(buy_plot, type='scatter', markersize=60, marker='^', color='red'))
        
        # 売りマーカー
        if sell_markers:
            sell_prices = [self.df['High'].iloc[i] * 1.02 for i in sell_markers]
            sell_plot = [np.nan] * len(self.df)
            for i, price in zip(sell_markers, sell_prices):
                sell_plot[i] = price
            addplots.append(mpf.make_addplot(sell_plot, type='scatter', markersize=60, marker='v', color='blue'))
        
        print(f"売買ポイント: 買い{len(buy_markers)}回, 売り{len(sell_markers)}回")
        
        # スタイル設定
        mc = mpf.make_marketcolors(up='r', down='b', inherit=True)
        s = mpf.make_mpf_style(base_mpf_style='default', marketcolors=mc)
        
        # チャート描画
        fig, axlist = mpf.plot(
            self.df,
            type='candle',
            style=s,
            addplot=addplots,
            figsize=(16, 8),
            title=f'\nTrading Simulation for {self.stock_code}',
            ylabel='Stock Price (JPY)',
            xrotation=0,
            datetime_format='%m/%d',
            panel_ratios=(6, 2, 2),
            returnfig=True,
            volume=True,
            volume_panel=0,
            ylabel_lower='Volume'
        )
        
        # パネルのラベル設定
        if len(axlist) > 2:
            axlist[2].set_ylabel('MACD')
        if len(axlist) > 4:
            axlist[4].set_ylabel('RSI')
        
        plt.show()
    
    def _plot_asset_history(self) -> None:
        """資産推移チャートの描画（内部メソッド）"""
        if self.asset_history:
            dates, assets = zip(*self.asset_history)
            fig_asset, ax_asset = plt.subplots(figsize=(16, 4))
            ax_asset.plot(dates, assets, label="資産推移", color="blue", linewidth=2)
            ax_asset.axhline(y=self.initial_cash, color='gray', linestyle='--', alpha=0.7, label='初期資金')
            ax_asset.set_title("資産の推移")
            ax_asset.set_ylabel("円")
            ax_asset.grid(True, alpha=0.3)
            ax_asset.legend()
            ax_asset.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
            plt.tight_layout()
            plt.show()
    
    def run_full_analysis(self) -> 'TradingSystem':
        """完全な分析を実行する（データ準備→シミュレーション→結果表示）"""
        print(f"🚀 TradingSystem分析開始 - 銘柄: {self.stock_code}")
        print("="*50)
        
        # 1. データ準備
        if self.prepare_data() is None:
            print("❌ データ準備に失敗しました。")
            return self
        
        # 2. シミュレーション実行
        asset_history, trade_history, final_cash = self.run_simulation()
        if asset_history is None:
            print("❌ シミュレーションに失敗しました。")
            return self
        
        # 3. 結果表示
        self.show_results()
        
        print("\n✅ 分析完了！")
        return self
