# ウォークフォワード分析 (Walk-Forward Analysis)

## 概要

ウォークフォワード分析は、トレーディングシステムのパラメータ最適化において、将来のデータを使った過学習（オーバーフィッティング）を防ぐための重要な手法です。

## 原理

### 従来の最適化の問題点

従来のバックテストでは、過去の全データを使って一度だけパラメータを最適化します。これには以下の問題があります：

1. **過学習（オーバーフィッティング）**: 特定の期間に特化したパラメータになりがち
2. **将来のデータ漏洩**: 最適化に将来のデータが含まれる可能性
3. **市場環境の変化**: 過去の最適パラメータが将来も有効とは限らない

### ウォークフォワード分析の解決策

ウォークフォワード分析では、以下の手順で分析を行います：

1. **学習期間**: 過去のデータでパラメータを最適化
2. **検証期間**: 最適化されたパラメータで将来のパフォーマンスを検証
3. **期間の移動**: 一定期間ずらして繰り返し実行
4. **統計的分析**: 全期間の結果を統計的に分析

## 実装詳細

### 期間設定

```python
training_period_years = 2      # 学習期間: 2年
validation_period_months = 6   # 検証期間: 6ヶ月
step_months = 6               # ステップ期間: 6ヶ月
```

### 分析フロー

```
期間1: [学習: 2020-2022] → [検証: 2022-2022.5]
期間2: [学習: 2020.5-2022.5] → [検証: 2022.5-2023]
期間3: [学習: 2021-2023] → [検証: 2023-2023.5]
...
```

### パフォーマンス指標

各検証期間で以下の指標を計算します：

- **総損益**: 検証期間の総損益
- **収益率**: 総損益 / 初期資金
- **プロフィットファクター**: 総利益 / 総損失
- **最大ドローダウン**: 最大の資産減少率
- **シャープレシオ**: リスク調整後収益率
- **勝率**: 利益取引の割合
- **取引回数**: 総取引回数

## 使用方法

### 基本的な使用方法

```python
from walk_forward_analysis import WalkForwardAnalyzer

# アナライザーの初期化
analyzer = WalkForwardAnalyzer(
    stock_code="7203.JP",
    initial_cash=1_000_000
)

# ウォークフォワード分析実行
results = analyzer.run_walk_forward_analysis(
    training_period_years=2,
    validation_period_months=6,
    step_months=6,
    n_trials=50
)

# 結果の可視化
analyzer.plot_results()

# 結果の保存
analyzer.save_results("walk_forward_results.csv")
```

### テスト用スクリプト

短時間でテストする場合は、`test_walk_forward.py`を使用します：

```bash
python examples/test_walk_forward.py
```

## 結果の解釈

### 統計指標

分析完了後、以下の統計指標が表示されます：

- **平均総損益**: 全期間の平均損益
- **平均収益率**: 全期間の平均収益率
- **平均プロフィットファクター**: 全期間の平均プロフィットファクター
- **期間勝率**: 利益を上げた期間の割合
- **標準偏差**: 結果のばらつき

### 可視化

以下の6つのグラフが生成されます：

1. **総損益の推移**: 各期間の総損益
2. **収益率の推移**: 各期間の収益率
3. **プロフィットファクターの推移**: 各期間のプロフィットファクター
4. **最大ドローダウンの推移**: 各期間の最大ドローダウン
5. **勝率の推移**: 各期間の勝率
6. **取引回数の推移**: 各期間の取引回数

## パラメータ最適化

### 最適化対象パラメータ

- **短期移動平均期間**: 3-20日
- **中期移動平均期間**: 15-50日
- **長期移動平均期間**: 40-200日

### 制約条件

```python
# 短期 < 中期 < 長期
if not (sma_short < sma_medium < sma_long):
    return float('-inf')
```

### 最適化アルゴリズム

OptunaのTPEサンプラーを使用して効率的な最適化を実行します。

## 出力ファイル

### CSVファイル

分析結果は以下の形式でCSVファイルに保存されます：

| 列名 | 説明 |
|------|------|
| period | 分析期間番号 |
| train_start | 学習開始日 |
| train_end | 学習終了日 |
| val_start | 検証開始日 |
| val_end | 検証終了日 |
| sma_short | 最適短期移動平均期間 |
| sma_medium | 最適中期移動平均期間 |
| sma_long | 最適長期移動平均期間 |
| total_profit | 総損益 |
| total_return | 収益率 |
| profit_factor | プロフィットファクター |
| max_drawdown | 最大ドローダウン |
| sharpe_ratio | シャープレシオ |
| win_rate | 勝率 |
| total_trades | 取引回数 |

## 注意事項

### データ要件

- 十分な長さの履歴データが必要（最低3-4年）
- データの品質と完全性を確認
- 欠損値や異常値の処理

### 計算時間

- 学習期間が長いほど計算時間が増加
- 最適化試行回数が多いほど精度が向上するが時間がかかる
- 並列処理の活用を検討

### 解釈の注意点

- 過去の結果は将来の保証ではない
- 市場環境の変化に注意
- 複数の銘柄での検証を推奨

## 応用例

### パラメータの安定性分析

```python
# パラメータの変化を分析
param_changes = []
for i in range(1, len(results)):
    prev_params = results[i-1]['best_params']
    curr_params = results[i]['best_params']
    
    change = {
        'period': i,
        'sma_short_change': curr_params['sma_short'] - prev_params['sma_short'],
        'sma_medium_change': curr_params['sma_medium'] - prev_params['sma_medium'],
        'sma_long_change': curr_params['sma_long'] - prev_params['sma_long']
    }
    param_changes.append(change)
```

### 市場環境別分析

```python
# 市場環境による分類
bull_market_results = []
bear_market_results = []

for result in results:
    if result['total_return'] > 0.1:  # 10%以上の収益
        bull_market_results.append(result)
    elif result['total_return'] < -0.1:  # -10%以下の損失
        bear_market_results.append(result)
```

## 今後の拡張

### 予定機能

1. **複数銘柄対応**: ポートフォリオレベルでの分析
2. **リスク指標の追加**: VaR、CVaRなどの高度なリスク指標
3. **機械学習統合**: MLモデルのパラメータ最適化
4. **リアルタイム分析**: 継続的なパフォーマンス監視

### 改善点

1. **並列処理**: 計算速度の向上
2. **メモリ最適化**: 大規模データセット対応
3. **インタラクティブ可視化**: より詳細な分析機能 