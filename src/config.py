# ==============================================================================
# --- 設定ファイル (config.py) ---
# ==============================================================================

# 銘柄コード（'XXXX.JP'形式）
STOCK_CODE = "7203.JP"  # 例：トヨタ自動車

# データ取得期間
DATA_PERIOD_DAYS = 300

# シミュレーション設定
INITIAL_CASH = 1_000_000  # 初期資金
INVESTMENT_RATIO = 0.5    # 1回の取引に使う資金の割合
FEE_RATE = 0.005          # 売買手数料率
SLIPPAGE_RATE = 0.0005    # スリッページ率

# ボラティリティ適応型リスク管理設定
TAKE_PROFIT_ATR_MULTIPLE = 3.0   # 利食い幅（ATRの倍数）
STOP_LOSS_ATR_MULTIPLE = 1.5     # 損切り幅（ATRの倍数）

# 従来の固定利食い・損切り設定（ATR設定が優先される）
TAKE_PROFIT_RATE = 0.10   # 10%上昇で利食い（ATR設定が0の場合に使用）
STOP_LOSS_RATE = 0.05     # 5%下落で損切り（ATR設定が0の場合に使用）

# 売買シグナルの閾値
BUY_THRESHOLD = 2.0
SELL_THRESHOLD = 2.0

# シグナルの重み設定
SIGNAL_WEIGHTS = {
    'golden_cross_short': 1.5,      # 短期ゴールデンクロス
    'golden_cross_long': 2.0,       # 長期ゴールデンクロス
    'bb_oversold': 1.0,             # ボリンジャーバンド下限突破
    'rsi_oversold': 1.2,            # RSI売られすぎ
    'macd_bullish': 1.3,            # MACDブリッシュクロス
    'double_bottom': 2.5,           # ダブルボトム
    'three_white_soldiers': 2.0,    # 赤三兵
    'three_gap_down': 1.8,          # 三空叩き込み
    'dead_cross_short': 1.5,        # 短期デッドクロス
    'dead_cross_long': 2.0,         # 長期デッドクロス
    'bb_overbought': 1.0,           # ボリンジャーバンド上限突破
    'rsi_overbought': 1.2,          # RSI買われすぎ
    'macd_bearish': 1.3,            # MACDベアリッシュクロス
    'double_top': 2.5,              # ダブルトップ
    'three_black_crows': 2.0,       # 黒三兵
    'three_gap_up': 1.8,            # 三空踏み上げ
}
