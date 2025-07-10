try:
    import streamlit as st  # type: ignore[reportMissingImports]
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError as e:
    missing = str(e).split("No module named ")[-1].replace("'", "")
    print(f"必要なパッケージ({missing})がインストールされていません。\n次のコマンドでインストールしてください: pip install {missing}")
    import sys
    sys.exit(1)
from src.trading_system import TradingSystem

st.set_page_config(page_title="TradingSystemPro Dashboard", layout="wide")

@st.cache_data(show_spinner=False)
def run_analysis():
    system = TradingSystem()
    system.run_full_analysis()
    return system

st.title("📈 TradingSystemPro パフォーマンスダッシュボード")

# 定時実行（例: 1日1回）を簡易的に再現
if st.button("最新データで分析を実行（手動更新）"):
    st.cache_data.clear()
    st.success("分析を再実行しました。")

system = run_analysis()

# 資産推移グラフ
st.header("資産推移")
if system.asset_history:
    # asset_historyがlist of dictならcolumns指定不要
    df_asset = pd.DataFrame(system.asset_history)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_asset["date"], df_asset["asset"], label="資産推移", color="blue")
    ax.set_xlabel("日付")
    ax.set_ylabel("資産(円)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
else:
    st.warning("資産推移データがありません。")

# 取引履歴
st.header("取引履歴")
if system.trade_history:
    df_trades = pd.DataFrame(system.trade_history)
    st.dataframe(df_trades)
else:
    st.warning("取引履歴がありません。")

# パフォーマンスサマリー
st.header("パフォーマンスサマリー")
final_cash = getattr(system, "final_cash", None)
initial_cash = getattr(system, "initial_cash", None)
if final_cash is not None and initial_cash is not None:
    total_profit = final_cash - initial_cash
    st.metric("最終資産", f"{final_cash:,.0f} 円")
    st.metric("総損益", f"{total_profit:,.0f} 円")
    st.metric("総収益率", f"{total_profit/initial_cash:.2%}")
else:
    st.info("パフォーマンスサマリーがありません。")

st.caption("※ 本ダッシュボードは毎日定時に自動更新する設計です。運用時はcronやWindowsタスクスケジューラ等で自動実行してください。") 