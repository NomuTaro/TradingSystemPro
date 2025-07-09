# TradingSystem Pro 📈

高度な株式取引分析・シミュレーションシステム

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-beta-orange.svg)](https://github.com/your-username/TradingSystemPro)

## 🌟 特徴

TradingSystem Proは、テクニカル分析、酒田五法、機械学習を統合した包括的な株式取引システムです。

### 🔥 主な機能

- **📊 基本的なテクニカル分析**: SMA、RSI、MACD、ボリンジャーバンド、ATR
- **🏮 酒田五法パターン検出**: ダブルトップ/ボトム、三山三川、赤三兵/黒三兵
- **🚀 複数銘柄並列分析**: ThreadPoolExecutorによる高速並列処理
- **⚡ 拡張テクニカル指標**: 一目均衡表、ストキャスティクス、CCI、ADX等
- **🤖 機械学習統合**: ランダムフォレスト・勾配ブースティング
- **🔧 パラメータ最適化**: グリッドサーチによる最適パラメータ探索
- **⏰ リアルタイムシミュレーション**: マルチスレッドによるリアルタイム取引

## 🚀 クイックスタート

### インストール

```bash
# 基本インストール
pip install -r requirements.txt

# 機械学習機能を含む完全インストール
pip install -r requirements.txt
pip install scikit-learn joblib

# 開発者向けインストール
pip install -e .[dev,ml,notebook]
```

### 基本的な使用方法

```python
from src import TradingSystem

# 基本的なトレーディングシステム
trading_system = TradingSystem()
trading_system.run_full_analysis()
```

### 拡張機能の使用例

```python
from src import (
    MultiStockAnalyzer,
    EnhancedTradingSystem,
    MLEnhancedTradingSystem,
    ParameterOptimizer,
    RealTimeTradingSimulator
)

# 複数銘柄並列分析
analyzer = MultiStockAnalyzer(['7203.JP', '6758.JP', '8306.JP'])
results = analyzer.run_parallel_analysis()
analyzer.compare_performance(results)

# 機械学習統合システム
ml_system = MLEnhancedTradingSystem()
ml_system.prepare_enhanced_data()
ml_system.prepare_ml_features()
ml_system.train_ml_model('random_forest')

# パラメータ最適化
optimizer = ParameterOptimizer('7203.JP')
param_grid = {
    'BUY_THRESHOLD': [0.4, 0.5, 0.6],
    'SELL_THRESHOLD': [0.4, 0.5, 0.6]
}
best_result, all_results = optimizer.optimize_parameters(param_grid)

# リアルタイムシミュレーション
simulator = RealTimeTradingSimulator(initial_cash=1000000)
simulator.start_simulation(base_price=3000, duration_minutes=5)
```

## 📁 プロジェクト構成

```
TradingSystemPro/
├── src/                          # メインパッケージ
│   ├── __init__.py              # パッケージ初期化
│   ├── config.py                # 設定ファイル
│   ├── trading_system.py        # メインのトレーディングシステム
│   └── extensions.py            # 拡張機能クラス群
├── notebooks/                   # Jupyter Notebook
│   └── TradingSystemPro_Demo.ipynb
├── tests/                       # テストファイル
├── docs/                        # ドキュメント
├── examples/                    # 使用例
├── requirements.txt             # 依存ライブラリ
├── setup.py                     # パッケージセットアップ
└── README.md                    # このファイル
```

## 🛠️ 設定

`src/config.py`ファイルで各種設定を変更できます：

```python
# 銘柄設定
STOCK_CODE = "7203.JP"  # 分析対象銘柄

# 資金設定
INITIAL_CASH = 1_000_000  # 初期資金
INVESTMENT_RATIO = 0.5    # 投資比率

# リスク管理設定
TAKE_PROFIT_ATR_MULTIPLE = 3.0  # 利食い幅（ATRの倍数）
STOP_LOSS_ATR_MULTIPLE = 1.5    # 損切り幅（ATRの倍数）

# シグナル設定
BUY_THRESHOLD = 2.0   # 買いシグナル閾値
SELL_THRESHOLD = 2.0  # 売りシグナル閾値
```

## 📊 出力ファイル

システムは以下のファイルを生成します：

- `multi_stock_analysis_results.csv` - 複数銘柄分析結果
- `ml_trading_model.pkl` - 機械学習モデル
- `ml_scaler.pkl` - 特徴量スケーラー
- `optimization_results.csv` - パラメータ最適化結果
- `realtime_trades.csv` - リアルタイム取引履歴

## 🧪 テスト

```bash
# テスト実行
pytest tests/

# カバレッジ付きテスト
pytest --cov=src tests/
```

## 📈 パフォーマンス指標

システムは以下の指標でパフォーマンスを評価します：

- **総リターン**: 投資期間中の総収益率
- **シャープレシオ**: リスク調整後リターン
- **最大ドローダウン**: 最大損失幅
- **勝率**: 勝ちトレードの割合
- **プロフィットファクター**: 総利益/総損失

## 🤝 貢献

プロジェクトへの貢献を歓迎します！

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルをご覧ください。

## ⚠️ 免責事項

このソフトウェアは教育・研究目的で提供されています。実際の投資判断にご利用される場合は、自己責任でお願いします。投資にはリスクが伴い、元本保証はありません。

## 🆘 サポート

- 📚 [ドキュメント](docs/)
- 🐛 [Issue報告](https://github.com/your-username/TradingSystemPro/issues)
- 💬 [ディスカッション](https://github.com/your-username/TradingSystemPro/discussions)

## 🙏 謝辞

このプロジェクトは以下のライブラリに依存しています：

- [pandas](https://pandas.pydata.org/) - データ分析
- [numpy](https://numpy.org/) - 数値計算
- [matplotlib](https://matplotlib.org/) - 可視化
- [mplfinance](https://github.com/matplotlib/mplfinance) - 金融チャート
- [pandas-ta](https://github.com/twopirllc/pandas-ta) - テクニカル分析
- [scikit-learn](https://scikit-learn.org/) - 機械学習

---

⭐ このプロジェクトが役に立った場合は、スターをお願いします！
