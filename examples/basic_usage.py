"""
TradingSystem Pro 基本使用例

このスクリプトでは、TradingSystem Proの基本的な機能を
実際のコードで示します。
"""

import sys
import os

# パッケージを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import TradingSystem, config

def basic_trading_example():
    """基本的なトレーディングシステムの使用例"""
    print("🚀 基本的なトレーディングシステムの使用例")
    print("=" * 50)
    
    # 1. TradingSystemインスタンス作成
    trading_system = TradingSystem()
    
    # 2. データ準備
    print("📊 データ準備中...")
    if trading_system.prepare_data() is None:
        print("❌ データ取得に失敗しました")
        return
    
    print(f"✅ データ取得完了 - {len(trading_system.df)}日分のデータ")
    
    # 3. シミュレーション実行
    print("🔄 シミュレーション実行中...")
    results = trading_system.run_simulation()
    
    if not results:
        print("❌ シミュレーションに失敗しました")
        return
    
    # 4. 結果表示
    print("📈 結果:")
    print(f"  銘柄コード: {trading_system.stock_code}")
    print(f"  初期資金: {config.INITIAL_CASH:,}円")
    print(f"  最終資金: {results['final_cash']:,.0f}円")
    print(f"  総リターン: {results['total_return_pct']:.2f}%")
    print(f"  取引回数: {results['total_trades']}回")
    print(f"  勝率: {results['win_rate']:.1f}%")
    print(f"  シャープレシオ: {results['sharpe_ratio']:.3f}")
    
    # 5. 詳細結果表示（オプション）
    # trading_system.show_results()
    
    print("\n✅ 基本例完了！")

def custom_parameters_example():
    """カスタムパラメータでの使用例"""
    print("\n🔧 カスタムパラメータでの使用例")
    print("=" * 50)
    
    # 設定を一時的に変更
    original_threshold = config.BUY_THRESHOLD
    original_cash = config.INITIAL_CASH
    
    try:
        # パラメータ変更
        config.BUY_THRESHOLD = 1.5  # より敏感な買いシグナル
        config.INITIAL_CASH = 500000  # 初期資金を半分に
        
        print(f"📊 設定変更:")
        print(f"  買い閾値: {config.BUY_THRESHOLD}")
        print(f"  初期資金: {config.INITIAL_CASH:,}円")
        
        # システム実行
        trading_system = TradingSystem()
        if trading_system.prepare_data() is not None:
            results = trading_system.run_simulation()
            
            if results:
                print(f"📈 カスタム設定結果:")
                print(f"  総リターン: {results['total_return_pct']:.2f}%")
                print(f"  取引回数: {results['total_trades']}回")
                print(f"  勝率: {results['win_rate']:.1f}%")
        
    finally:
        # 設定を元に戻す
        config.BUY_THRESHOLD = original_threshold
        config.INITIAL_CASH = original_cash
    
    print("✅ カスタムパラメータ例完了！")

def multiple_stocks_comparison():
    """複数銘柄での比較例"""
    print("\n📊 複数銘柄での比較例")
    print("=" * 50)
    
    # 日本の主要銘柄例
    stock_codes = ['7203.JP', '6758.JP', '8306.JP']
    results_summary = []
    
    original_stock_code = config.STOCK_CODE
    
    try:
        for stock_code in stock_codes:
            print(f"📈 分析中: {stock_code}")
            
            # 銘柄コード変更
            config.STOCK_CODE = stock_code
            
            # システム実行
            trading_system = TradingSystem()
            if trading_system.prepare_data() is not None:
                results = trading_system.run_simulation()
                
                if results:
                    results_summary.append({
                        'stock_code': stock_code,
                        'total_return_pct': results['total_return_pct'],
                        'win_rate': results['win_rate'],
                        'total_trades': results['total_trades']
                    })
    
    finally:
        # 設定を元に戻す
        config.STOCK_CODE = original_stock_code
    
    # 結果比較
    if results_summary:
        print("\n📋 比較結果:")
        print("-" * 60)
        print(f"{'銘柄':^10} {'リターン':^10} {'勝率':^8} {'取引数':^8}")
        print("-" * 60)
        
        for result in results_summary:
            print(f"{result['stock_code']:^10} "
                  f"{result['total_return_pct']:^8.2f}% "
                  f"{result['win_rate']:^6.1f}% "
                  f"{result['total_trades']:^8}")
        
        # 最高パフォーマンス
        best_stock = max(results_summary, key=lambda x: x['total_return_pct'])
        print(f"\n🏆 最高パフォーマンス: {best_stock['stock_code']} "
              f"({best_stock['total_return_pct']:.2f}%)")
    
    print("✅ 複数銘柄比較例完了！")

if __name__ == "__main__":
    print("🌟 TradingSystem Pro 使用例集")
    print("=" * 60)
    
    try:
        # 基本例
        basic_trading_example()
        
        # カスタムパラメータ例
        custom_parameters_example()
        
        # 複数銘柄比較例（時間がかかる場合があります）
        # multiple_stocks_comparison()
        
        print(f"\n🎉 全ての例が完了しました！")
        print(f"💡 より詳細な機能は notebooks/TradingSystemPro_Demo.ipynb をご覧ください")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        print("💡 requirements.txt の依存関係を確認してください")
