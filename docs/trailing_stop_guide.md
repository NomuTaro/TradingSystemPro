# Trading System Pro - トレーリングストップ機能ガイド

## 概要

TradingSystemProに新しく追加されたATRベースのトレーリングストップ機能について説明します。この機能により、利益が出ているポジションのストップロスラインが価格の上昇に合わせて自動的に引き上げられ、利益を保護することができます。

## トレーリングストップとは

トレーリングストップは、ポジションが利益を上げている時に、ストップロスラインを価格の上昇に合わせて引き上げる機能です。これにより：

1. **利益の保護**: 一度上げられたストップロスラインは下がらないため、利益を確実に確保できます
2. **損失の最小化**: 価格が反転した場合でも、引き上げられたストップロスラインで損失を最小限に抑えられます
3. **自動化**: 手動でのストップロス調整が不要になり、感情的な判断を排除できます

## 実装詳細

### 1. 初期設定

ポジションを開始する際に、初期のトレーリングストップ価格が設定されます：

```python
# ATRベースの場合
trailing_stop_price = buy_price - (buy_atr * stop_loss_atr_multiple)

# 固定率の場合
trailing_stop_price = buy_price * (1 - stop_loss_rate)
```

### 2. トレーリングストップの更新

毎日の価格変動に応じて、トレーリングストップ価格が更新されます：

```python
# ATRベースの更新
if buy_atr > 0:
    new_trailing_stop = today_price - (buy_atr * stop_loss_atr_multiple)
    if new_trailing_stop > trailing_stop_price:
        trailing_stop_price = new_trailing_stop

# 固定率ベースの更新
else:
    new_trailing_stop = today_price * (1 - stop_loss_rate)
    if new_trailing_stop > trailing_stop_price:
        trailing_stop_price = new_trailing_stop
```

### 3. 売り判断の優先順位

トレーリングストップ機能では、以下の優先順位で売り判断が行われます：

1. **トレーリングストップ**: 価格がトレーリングストップラインを下回った場合
2. **シグナル売り**: 売りシグナルが発生した場合
3. **利食い**: 利食いラインに到達した場合

## 設定パラメータ

### ATRベースの設定

```python
# config.py または TradingSystem インスタンスで設定
stop_loss_atr_multiple = 1.5  # ATRの1.5倍
take_profit_atr_multiple = 3.0  # ATRの3倍
```

### 固定率ベースの設定

```python
stop_loss_rate = 0.05  # 5%の損切り
take_profit_rate = 0.10  # 10%の利食い
```

## 使用例

### 基本的な使用方法

```python
from trading_system import TradingSystem

# TradingSystemインスタンスを作成
system = TradingSystem(stock_code="7203.JP")

# トレーリングストップ設定（デフォルトで有効）
system.stop_loss_atr_multiple = 1.5
system.take_profit_atr_multiple = 3.0

# シミュレーション実行
asset_history, trade_history, final_cash = system.run_simulation()
```

### トレーリングストップの無効化

```python
# 固定ストップロスのみを使用する場合
system.stop_loss_atr_multiple = 0  # ATRベースを無効化
system.stop_loss_rate = 0.05  # 固定率のみ使用
```

## 取引履歴の分析

トレーリングストップ機能では、取引履歴に以下の情報が追加されます：

### 買い取引
```python
{
    'type': 'BUY',
    'date': datetime,
    'price': float,
    'qty': int,
    'signal_score': float,
    'cost': float,
    'initial_stop': float  # 初期トレーリングストップ価格
}
```

### 売り取引
```python
{
    'type': 'SELL',
    'date': datetime,
    'price': float,
    'qty': int,
    'reason': str,  # 'Trailing Stop (price)' など
    'proceeds': float,
    'final_stop': float  # 最終トレーリングストップ価格
}
```

## テスト方法

### 1. 基本的なテスト

```bash
cd examples
python test_trailing_stop.py
```

このスクリプトは以下を実行します：
- トレーリングストップ機能のテスト
- 固定ストップロスとの比較
- 詳細な分析結果の表示

### 2. 手動でのテスト

```python
from trading_system import TradingSystem

# システムを作成
system = TradingSystem("7203.JP")
system.initial_cash = 1_000_000

# データ準備
df = system.prepare_data()

# シミュレーション実行
asset_history, trade_history, final_cash = system.run_simulation()

# トレーリングストップ取引の分析
trailing_stop_trades = []
for i, trade in enumerate(trade_history):
    if trade['type'] == 'BUY' and i + 1 < len(trade_history):
        sell_trade = trade_history[i + 1]
        if 'Trailing Stop' in sell_trade.get('reason', ''):
            trailing_stop_trades.append({
                'buy_price': trade['price'],
                'sell_price': sell_trade['price'],
                'initial_stop': trade['initial_stop'],
                'final_stop': sell_trade['final_stop']
            })

print(f"トレーリングストップ取引数: {len(trailing_stop_trades)}")
```

## パフォーマンス分析

### トレーリングストップの効果測定

```python
# トレーリングストップ取引の分析
trailing_profits = []
for trade in trailing_stop_trades:
    profit = trade['sell_price'] - trade['buy_price']
    trailing_profits.append(profit)

# 統計情報
avg_profit = np.mean(trailing_profits)
win_rate = len([p for p in trailing_profits if p > 0]) / len(trailing_profits) * 100

print(f"平均利益: {avg_profit:,.0f}円")
print(f"勝率: {win_rate:.1f}%")
```

## 注意事項

### 1. オーバーフィッティングのリスク

トレーリングストップ機能は過去データに最適化されている可能性があります。将来のパフォーマンスを保証するものではありません。

### 2. 市場環境への依存

トレーリングストップの効果は市場環境に大きく依存します：
- **トレンド市場**: 効果的
- **レンジ市場**: 効果が限定的
- **ボラティリティの高い市場**: 頻繁なストップアウトの可能性

### 3. パラメータの調整

最適なパラメータは銘柄や期間によって異なります：
- ATR倍数の調整
- 固定率の調整
- 利食い設定との組み合わせ

## 高度な設定

### 1. 動的ATR倍数

```python
# 市場ボラティリティに応じてATR倍数を調整
def get_dynamic_atr_multiple(current_atr, historical_atr):
    if current_atr > historical_atr * 1.5:
        return 2.0  # 高ボラティリティ時は緩い設定
    else:
        return 1.5  # 通常時は標準設定
```

### 2. 部分決済との組み合わせ

```python
# 利益が出た場合の部分決済
if current_profit > buy_price * 0.05:  # 5%以上の利益
    # ポジションの50%を決済
    partial_sell_qty = position * 0.5
    # 残りのポジションでトレーリングストップ継続
```

### 3. 時間ベースの調整

```python
# 保有期間に応じたストップロス調整
holding_days = (current_date - buy_date).days
if holding_days > 30:
    # 長期保有時はより緩いストップロス
    trailing_stop_multiple = 2.0
else:
    # 短期保有時は標準設定
    trailing_stop_multiple = 1.5
```

## トラブルシューティング

### 1. トレーリングストップが動作しない

- ATRデータが正しく計算されているか確認
- パラメータ設定を確認
- デバッグログを有効化

### 2. 頻繁なストップアウト

- ATR倍数を大きくする
- 固定率を緩くする
- 市場環境を確認

### 3. 利益が十分に確保できない

- 利食い設定を調整
- トレーリングストップの開始タイミングを調整
- 部分決済の検討

## まとめ

トレーリングストップ機能は、利益の保護と損失の最小化を自動化する強力なツールです。適切なパラメータ設定と市場環境の理解により、より効果的なリスク管理が可能になります。

定期的なバックテストとパフォーマンス分析を行い、市場環境に応じてパラメータを調整することをお勧めします。 