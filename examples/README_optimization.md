# Trading System Pro - Optuna最適化機能

このディレクトリには、TradingSystemProの移動平均線期間をOptunaを使って最適化する機能が含まれています。

## 概要

Optunaは、ハイパーパラメータ最適化のためのPythonライブラリです。この実装では、TradingSystemの移動平均線期間（短期、中期、長期）を最適化して、総損益を最大化します。

## ファイル構成

- `optimize_parameters.py` - 完全な最適化機能（可視化含む）
- `test_optimization.py` - シンプルな最適化テスト
- `README_optimization.md` - このファイル

## インストール

### 1. Optunaのインストール

```bash
pip install optuna
```

または、requirements.txtを使用：

```bash
pip install -r requirements.txt
```

### 2. 依存関係の確認

以下のライブラリが必要です：
- optuna >= 3.0.0
- pandas >= 1.5.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- mplfinance >= 0.12.0
- pandas-datareader >= 0.10.0

## 使用方法

### 1. シンプルな最適化テスト

```bash
cd examples
python test_optimization.py
```

このスクリプトは：
- 20回の試行で最適化を実行
- 最適パラメータでのバックテストを実行
- デフォルトパラメータとの比較を表示

### 2. 完全な最適化機能

```bash
cd examples
python optimize_parameters.py
```

このスクリプトは：
- 50回の試行で最適化を実行
- 詳細な可視化を提供
- 最適化履歴の分析
- 3D散布図での結果表示

## 最適化パラメータ

### 移動平均線期間

- **短期移動平均**: 3-20日
- **中期移動平均**: 15-50日  
- **長期移動平均**: 40-200日

### 制約条件

- 短期 < 中期 < 長期（期間の順序制約）

### 目的関数

総損益（最終資産 - 初期資金）を最大化

## 出力例

```
=== Trading System Pro - Optuna最適化テスト ===

銘柄: 7203.JP
初期資金: 1,000,000円
試行回数: 20
==================================================
最適化開始...
試行 0: SMA(5, 25, 75) -> 損益: 45,230円
試行 1: SMA(8, 30, 80) -> 損益: 52,180円
...

✅ 最適化完了!
最適パラメータ:
  短期移動平均: 7日
  中期移動平均: 28日
  長期移動平均: 85日
最適総損益: 78,450円

📊 最適パラメータでのバックテスト実行...

🎯 最適化バックテスト結果:
最終資産: 1,078,450円
総損益: 78,450円
総収益率: 7.85%
取引回数: 12回
勝率: 66.7%
```

## カスタマイズ

### パラメータ範囲の変更

`optimize_parameters.py`の`objective`関数内で、以下の部分を変更できます：

```python
# 移動平均線の期間をサンプリング
sma_short = trial.suggest_int('sma_short', 3, 20, step=1)      # 短期移動平均（3-20日）
sma_medium = trial.suggest_int('sma_medium', 15, 50, step=1)   # 中期移動平均（15-50日）
sma_long = trial.suggest_int('sma_long', 40, 200, step=1)      # 長期移動平均（40-200日）
```

### 試行回数の変更

```python
n_trials = 100  # より多くの試行で精度を向上
```

### 銘柄の変更

```python
stock_code = "6758.JP"  # ソニー
# または
stock_code = "9984.JP"  # ソフトバンクグループ
```

## 注意事項

1. **計算時間**: 試行回数を増やすと計算時間が長くなります
2. **データ取得**: インターネット接続が必要です
3. **最適化の限界**: 過去データでの最適化は将来のパフォーマンスを保証しません
4. **オーバーフィッティング**: 過度な最適化はオーバーフィッティングを引き起こす可能性があります

## トラブルシューティング

### Optunaがインストールされていない場合

```bash
pip install optuna
```

### データ取得エラー

- インターネット接続を確認
- 銘柄コードが正しいか確認
- データプロバイダーの制限を確認

### メモリ不足

- 試行回数を減らす
- データ期間を短縮する
- より軽量なパラメータ範囲を使用する

## 高度な機能

### 複数銘柄での最適化

```python
stock_codes = ["7203.JP", "6758.JP", "9984.JP"]
for stock_code in stock_codes:
    optimizer = TradingSystemOptimizer(stock_code=stock_code)
    results = optimizer.optimize(n_trials=50)
```

### 異なる評価指標での最適化

現在は総損益を最大化していますが、以下の指標に変更することも可能です：

- シャープレシオ
- 最大ドローダウン
- 勝率
- リスク調整後リターン

### 並列処理

Optunaは並列処理をサポートしています：

```python
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=100, n_jobs=-1)  # 全CPUコアを使用
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。 