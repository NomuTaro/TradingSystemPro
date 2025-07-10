# ==============================================================================
# --- 機械学習統合テストスクリプト ---
# ==============================================================================

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from trading_system import TradingSystem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def test_ml_integration():
    """機械学習統合機能のテスト"""
    print("🤖 機械学習統合テスト開始")
    print("="*50)
    
    # TradingSystemの初期化
    ts = TradingSystem("7203.JP")  # トヨタ自動車
    
    # データ準備（MLモデルも自動的に訓練される）
    print("\n📊 データ準備中...")
    df = ts.prepare_data()
    
    if df is None:
        print("❌ データ準備に失敗しました。")
        return
    
    print(f"✅ データ準備完了 - {len(df)}行のデータ")
    
    # MLモデルの状態確認
    print(f"\n🔍 MLモデル状態:")
    print(f"  モデル: {'✅ 訓練済み' if ts.ml_model is not None else '❌ 未訓練'}")
    print(f"  スケーラー: {'✅ 準備済み' if ts.scaler is not None else '❌ 未準備'}")
    print(f"  特徴量数: {len(ts.feature_columns)}")
    print(f"  予測期間: {ts.prediction_horizon}日後")
    
    # 最新データでの予測テスト
    if ts.ml_model is not None:
        print("\n🔮 最新データでの予測テスト:")
        latest_data = df.iloc[-1:]
        up_prob, down_prob = ts._predict_price_movement(latest_data)
        print(f"  上昇確率: {up_prob:.3f} ({up_prob*100:.1f}%)")
        print(f"  下落確率: {down_prob:.3f} ({down_prob*100:.1f}%)")
        print(f"  予測: {'📈 上昇' if up_prob > down_prob else '📉 下落'}")
    
    # シグナル評価のテスト
    print("\n📈 シグナル評価テスト:")
    if len(df) >= 100:
        # 過去100日分のデータでシグナル評価
        test_data = df.iloc[:100]
        buy_score = ts.evaluate_buy_signals(test_data, 100)
        sell_score = ts.evaluate_sell_signals(test_data, 100)
        
        print(f"  買いシグナルスコア: {buy_score:.3f}")
        print(f"  売りシグナルスコア: {sell_score:.3f}")
        print(f"  買い判断: {'✅' if buy_score >= ts.buy_threshold else '❌'}")
        print(f"  売り判断: {'✅' if sell_score >= ts.sell_threshold else '❌'}")
    
    # シミュレーション実行
    print("\n🚀 シミュレーション実行中...")
    asset_history, trade_history, final_cash = ts.run_simulation()
    
    if asset_history is None:
        print("❌ シミュレーションに失敗しました。")
        return
    
    # 結果表示
    print("\n📊 シミュレーション結果:")
    print(f"  最終資産: {final_cash:,.0f}円")
    print(f"  総損益: {final_cash - ts.initial_cash:,.0f}円")
    print(f"  取引回数: {len([t for t in trade_history if t['type'] == 'BUY'])}回")
    
    # ML予測の精度分析
    if ts.ml_model is not None and 'target' in df.columns:
        analyze_ml_accuracy(df, ts)
    
    # 結果表示
    ts.show_results()

def analyze_ml_accuracy(df: pd.DataFrame, ts: TradingSystem):
    """ML予測の精度を分析"""
    print("\n📊 ML予測精度分析:")
    
    # 実際の価格変動と予測を比較
    if 'target' in df.columns and ts.ml_model is not None:
        # テストデータでの予測
        test_data = df[ts.feature_columns + ['target']].dropna()
        if len(test_data) > 0:
            X_test = test_data[ts.feature_columns]
            y_test = test_data['target']
            
            # 予測実行
            X_test_scaled = ts.scaler.transform(X_test)
            y_pred = ts.ml_model.predict(X_test_scaled)
            y_pred_proba = ts.ml_model.predict_proba(X_test_scaled)
            
            # 精度計算
            accuracy = (y_pred == y_test).mean()
            print(f"  予測精度: {accuracy:.3f} ({accuracy*100:.1f}%)")
            
            # 信頼度別の精度
            confidence_thresholds = [0.6, 0.7, 0.8, 0.9]
            for threshold in confidence_thresholds:
                high_conf_mask = (y_pred_proba.max(axis=1) >= threshold)
                if high_conf_mask.sum() > 0:
                    high_conf_accuracy = (y_pred[high_conf_mask] == y_test[high_conf_mask]).mean()
                    print(f"  信頼度{threshold*100:.0f}%以上での精度: {high_conf_accuracy:.3f} ({high_conf_mask.sum()}サンプル)")

def compare_with_traditional():
    """従来手法との比較"""
    print("\n⚖️ 従来手法との比較:")
    
    # ML統合版
    ts_ml = TradingSystem("7203.JP")
    df_ml = ts_ml.prepare_data()
    if df_ml is not None:
        asset_history_ml, trade_history_ml, final_cash_ml = ts_ml.run_simulation()
        profit_ml = final_cash_ml - ts_ml.initial_cash if final_cash_ml else 0
    
    # 従来版（ML重みを0に設定）
    ts_traditional = TradingSystem("7203.JP")
    ts_traditional.signal_weights['ml_prediction'] = 0.0  # ML予測を無効化
    df_traditional = ts_traditional.prepare_data()
    if df_traditional is not None:
        asset_history_traditional, trade_history_traditional, final_cash_traditional = ts_traditional.run_simulation()
        profit_traditional = final_cash_traditional - ts_traditional.initial_cash if final_cash_traditional else 0
    
    if 'profit_ml' in locals() and 'profit_traditional' in locals():
        print(f"  ML統合版の損益: {profit_ml:,.0f}円")
        print(f"  従来版の損益: {profit_traditional:,.0f}円")
        if profit_ml > profit_traditional:
            print("  ✅ ML統合版が優れています")
        else:
            print("  ❌ 従来版の方が良い結果でした")

def plot_ml_predictions(df: pd.DataFrame, ts: TradingSystem):
    """ML予測の可視化"""
    if ts.ml_model is None or 'target' not in df.columns:
        return
    
    print("\n📈 ML予測の可視化:")
    
    # 予測確率を計算
    predictions = []
    for i in range(len(df)):
        if i < 75:  # 最低限必要なデータ数
            predictions.append((0.5, 0.5))
            continue
        
        df_hist = df.iloc[:i+1]
        up_prob, down_prob = ts._predict_price_movement(df_hist)
        predictions.append((up_prob, down_prob))
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # 価格チャート
    ax1.plot(df.index, df['Close'], label='株価', color='blue', linewidth=1)
    ax1.set_title('株価とML予測')
    ax1.set_ylabel('株価 (円)')
    ax1.grid(True, alpha=0.3)
    
    # 予測確率
    up_probs = [p[0] for p in predictions]
    down_probs = [p[1] for p in predictions]
    
    ax2.plot(df.index, up_probs, label='上昇確率', color='green', linewidth=1)
    ax2.plot(df.index, down_probs, label='下落確率', color='red', linewidth=1)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title('ML予測確率')
    ax2.set_ylabel('確率')
    ax2.set_xlabel('日付')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        test_ml_integration()
        print("\n✅ 機械学習統合テスト完了")
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc() 