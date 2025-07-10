# 機械学習統合ガイド

## 概要

TradingSystemProにRandomForestClassifierを使用した機械学習機能を統合しました。この機能により、テクニカル指標を特徴量として5日後の価格変動（上昇/下落）を予測し、従来のテクニカル分析と組み合わせてより精度の高い売買シグナルを生成します。

## 主な機能

### 1. 機械学習モデル
- **アルゴリズム**: RandomForestClassifier
- **予測対象**: 5日後の価格が上昇するか下落するか（二値分類）
- **特徴量**: 14種類のテクニカル指標
- **自動訓練**: データ準備時に自動的にモデルが訓練される

### 2. 特徴量（テクニカル指標）
以下のテクニカル指標を特徴量として使用：

- **移動平均**: SMA5, SMA25, SMA75
- **RSI**: 14日RSI
- **MACD**: MACD, MACD_signal, MACD_histogram
- **ボリンジャーバンド**: BB_upper, BB_middle, BB_lower
- **ATR**: 14日ATR
- **ADX**: ADX, PLUS_DI, MINUS_DI
- **出来高**: Volume

### 3. シグナル統合
- ML予測結果を従来のテクニカルシグナルと組み合わせ
- 設定可能な重み付け（デフォルト: 2.0）
- 買いシグナル: 上昇確率が高い場合にプラス
- 売りシグナル: 下落確率が高い場合にプラス

## 使用方法

### 基本的な使用例

```python
from src.trading_system import TradingSystem

# TradingSystemの初期化
ts = TradingSystem("7203.JP")

# データ準備（MLモデルも自動的に訓練される）
df = ts.prepare_data()

# シミュレーション実行
asset_history, trade_history, final_cash = ts.run_simulation()

# 結果表示
ts.show_results()
```

### ML予測の確認

```python
# 最新データでの予測
latest_data = df.iloc[-1:]
up_prob, down_prob = ts._predict_price_movement(latest_data)
print(f"上昇確率: {up_prob:.3f}")
print(f"下落確率: {down_prob:.3f}")
```

### ML重みの調整

```python
# ML予測の重みを調整
ts.signal_weights['ml_prediction'] = 3.0  # より重視
ts.signal_weights['ml_prediction'] = 0.0  # 無効化
```

## 設定項目

### config.pyでの設定

```python
CONFIG = {
    # ... 既存の設定 ...
    
    'SIGNAL_WEIGHTS': {
        # ... 既存のシグナル重み ...
        'ml_prediction': 2.0,  # 機械学習予測の重み
    }
}
```

### 推奨設定

- **保守的**: `ml_prediction: 1.0` - テクニカル分析を重視
- **バランス**: `ml_prediction: 2.0` - デフォルト設定
- **積極的**: `ml_prediction: 3.0` - ML予測を重視

## モデル性能

### 評価指標
- **精度**: テストデータでの正解率
- **特徴量重要度**: 各指標の予測への貢献度
- **信頼度別精度**: 予測確率に基づく精度

### 性能向上のポイント
1. **十分なデータ**: 最低100日以上のデータが必要
2. **特徴量の品質**: 欠損値のない高品質な指標
3. **定期的な再訓練**: 市場環境の変化に対応

## テストと検証

### テストスクリプトの実行

```bash
python examples/test_ml_integration.py
```

### 従来手法との比較

```python
# ML統合版
ts_ml = TradingSystem("7203.JP")
# 従来版（ML重みを0に設定）
ts_traditional = TradingSystem("7203.JP")
ts_traditional.signal_weights['ml_prediction'] = 0.0
```

## 注意事項

### 1. データ要件
- 最低100日以上のデータが必要
- 欠損値のない完全なデータセット
- 十分な特徴量（最低5種類以上）

### 2. 過学習の防止
- 時系列分割による検証
- 特徴量の正規化
- モデルパラメータの調整

### 3. 市場環境の変化
- 定期的なモデル再訓練の推奨
- 異なる市場環境での検証
- バックテストでの十分な検証

## トラブルシューティング

### よくある問題

1. **MLモデルが訓練されない**
   - データ量が不足している可能性
   - 特徴量が不足している可能性
   - 欠損値が多すぎる可能性

2. **予測精度が低い**
   - 特徴量の選択を見直す
   - モデルパラメータを調整する
   - より長期間のデータを使用する

3. **メモリ不足**
   - データ期間を短縮する
   - 特徴量を削減する
   - モデルパラメータを調整する

### デバッグ方法

```python
# MLモデルの状態確認
print(f"モデル: {ts.ml_model is not None}")
print(f"特徴量数: {len(ts.feature_columns)}")
print(f"データ行数: {len(ts.df) if ts.df is not None else 0}")

# 予測の詳細確認
if ts.ml_model is not None:
    print(f"特徴量重要度: {ts.ml_model.feature_importances_}")
```

## 今後の拡張予定

1. **複数のアルゴリズム**: XGBoost, LightGBM等の追加
2. **時系列特化モデル**: LSTM, GRU等のディープラーニング
3. **アンサンブル学習**: 複数モデルの組み合わせ
4. **自動ハイパーパラメータ調整**: Optunaとの統合
5. **リアルタイム予測**: ストリーミングデータ対応

## まとめ

機械学習統合により、従来のテクニカル分析に加えて、データ駆動型の予測機能を活用できるようになりました。適切な設定と検証により、より精度の高い売買シグナルを生成し、投資パフォーマンスの向上が期待できます。

ただし、機械学習モデルは過去のデータに基づいて学習するため、市場環境の変化には注意が必要です。定期的なモデル更新と十分なバックテストを実施して、安全に運用してください。 