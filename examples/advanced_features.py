"""
TradingSystem Pro 拡張機能使用例

このスクリプトでは、TradingSystem Proの拡張機能（並列分析、
機械学習、パラメータ最適化等）の使用方法を実際のコードで示します。
"""

import sys
import os

# パッケージを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import (
    MultiStockAnalyzer,
    EnhancedTradingSystem,
    MLEnhancedTradingSystem,
    ParameterOptimizer,
    RealTimeTradingSimulator,
    config
)

def multi_stock_analysis_example():
    """複数銘柄並列分析の使用例"""
    print("🚀 複数銘柄並列分析の使用例")
    print("=" * 50)
    
    # 分析対象銘柄（実際に存在する銘柄コードに変更してください）
    stock_codes = ['7203.JP', '6758.JP', '8306.JP']
    
    try:
        # MultiStockAnalyzer初期化
        analyzer = MultiStockAnalyzer(stock_codes, initial_cash=1000000)
        
        # 並列分析実行
        print(f"📊 {len(stock_codes)}銘柄の並列分析を開始...")
        results_df = analyzer.run_parallel_analysis(max_workers=2)
        
        if len(results_df) > 0:
            print("\n📈 分析結果:")
            print(results_df[['stock_code', 'total_return_pct', 'win_rate', 
                            'sharpe_ratio']].round(2))
            
            # パフォーマンス比較（チャート表示）
            # analyzer.compare_performance(results_df)
            
            print("\n🏆 最高パフォーマンス:")
            best_stock = results_df.iloc[0]
            print(f"  銘柄: {best_stock['stock_code']}")
            print(f"  リターン: {best_stock['total_return_pct']:.2f}%")
            print(f"  シャープレシオ: {best_stock['sharpe_ratio']:.3f}")
        else:
            print("❌ 分析結果が取得できませんでした")
    
    except Exception as e:
        print(f"❌ エラー: {e}")
    
    print("✅ 複数銘柄分析例完了！")

def enhanced_indicators_example():
    """拡張テクニカル指標の使用例"""
    print("\n⚡ 拡張テクニカル指標の使用例")
    print("=" * 50)
    
    try:
        # 拡張システム初期化
        enhanced_system = EnhancedTradingSystem()
        
        # 拡張データ準備
        print("📊 拡張テクニカル指標を計算中...")
        if enhanced_system.prepare_enhanced_data() is not None:
            
            # 最新データの指標値表示
            latest_data = enhanced_system.df.iloc[-1]
            
            print("\n📈 最新の拡張テクニカル指標:")
            print(f"  一目均衡表-転換線: {latest_data.get('tenkan_sen', 'N/A')}")
            print(f"  一目均衡表-基準線: {latest_data.get('kijun_sen', 'N/A')}")
            print(f"  ストキャスティクス%K: {latest_data.get('stoch_k', 'N/A')}")
            print(f"  ウィリアムズ%R: {latest_data.get('williams_r', 'N/A')}")
            print(f"  CCI: {latest_data.get('cci', 'N/A')}")
            print(f"  MFI: {latest_data.get('mfi', 'N/A')}")
            print(f"  ADX: {latest_data.get('adx', 'N/A')}")
            
            print(f"\n📊 拡張指標数: {len([col for col in enhanced_system.df.columns if any(x in col for x in ['tenkan', 'stoch', 'williams', 'cci', 'mfi', 'adx'])])}個")
        else:
            print("❌ データ取得に失敗しました")
    
    except Exception as e:
        print(f"❌ エラー: {e}")
    
    print("✅ 拡張指標例完了！")

def machine_learning_example():
    """機械学習統合の使用例"""
    print("\n🤖 機械学習統合の使用例")
    print("=" * 50)
    
    try:
        # ML拡張システム初期化
        ml_system = MLEnhancedTradingSystem()
        
        # データ準備
        print("📊 データと特徴量を準備中...")
        if ml_system.prepare_enhanced_data() is not None:
            ml_system.prepare_ml_features()
            
            print(f"✅ 特徴量数: {len(ml_system.feature_columns)}個")
            
            # モデル訓練（軽量版）
            print("🎯 機械学習モデルを訓練中...")
            ml_system.train_ml_model('random_forest')
            
            if ml_system.ml_model is not None:
                print("✅ 機械学習モデル訓練完了")
                print("💾 モデルが保存されました")
            else:
                print("❌ モデル訓練に失敗しました")
        else:
            print("❌ データ準備に失敗しました")
    
    except Exception as e:
        print(f"❌ エラー: {e}")
        print("💡 scikit-learnがインストールされていることを確認してください")
    
    print("✅ 機械学習例完了！")

def parameter_optimization_example():
    """パラメータ最適化の使用例"""
    print("\n🔧 パラメータ最適化の使用例")
    print("=" * 50)
    
    try:
        # 最適化システム初期化
        optimizer = ParameterOptimizer(config.STOCK_CODE)
        
        # 最適化するパラメータ範囲（小さなグリッドで高速化）
        param_grid = {
            'BUY_THRESHOLD': [1.5, 2.0],
            'SELL_THRESHOLD': [1.5, 2.0],
            'STOP_LOSS_RATE': [0.03, 0.05]
        }
        
        print(f"🔍 パラメータ最適化開始...")
        print(f"📊 {len(param_grid['BUY_THRESHOLD']) * len(param_grid['SELL_THRESHOLD']) * len(param_grid['STOP_LOSS_RATE'])}通りの組み合わせを評価")
        
        # 最適化実行
        best_result, all_results = optimizer.optimize_parameters(
            param_grid, 'total_return_pct'
        )
        
        if best_result:
            print(f"\n🏆 最適パラメータ:")
            print(f"  パラメータ: {best_result['parameters']}")
            print(f"  総リターン: {best_result['total_return_pct']:.2f}%")
            print(f"  シャープレシオ: {best_result['sharpe_ratio']:.3f}")
            print(f"  勝率: {best_result['win_rate']:.1f}%")
            
            print(f"\n📊 評価した組み合わせ数: {len(all_results)}")
        else:
            print("❌ 最適化結果が取得できませんでした")
    
    except Exception as e:
        print(f"❌ エラー: {e}")
    
    print("✅ パラメータ最適化例完了！")

def realtime_simulation_example():
    """リアルタイムシミュレーションの使用例"""
    print("\n⏰ リアルタイムシミュレーションの使用例")
    print("=" * 50)
    
    try:
        # リアルタイムシミュレーター初期化
        simulator = RealTimeTradingSimulator(initial_cash=1000000)
        
        print("🚀 リアルタイムシミュレーション開始（1分間）...")
        print("💡 価格変動と取引をリアルタイムで確認できます")
        
        # シミュレーション実行（短時間版）
        simulator.start_simulation(base_price=3000, duration_minutes=1)
        
        print("✅ リアルタイムシミュレーション完了")
    
    except Exception as e:
        print(f"❌ エラー: {e}")
    
    print("✅ リアルタイムシミュレーション例完了！")

if __name__ == "__main__":
    print("🌟 TradingSystem Pro 拡張機能使用例集")
    print("=" * 60)
    
    try:
        # 各拡張機能のデモ実行
        print("1️⃣ 複数銘柄並列分析")
        multi_stock_analysis_example()
        
        print("\n2️⃣ 拡張テクニカル指標")
        enhanced_indicators_example()
        
        print("\n3️⃣ 機械学習統合")
        machine_learning_example()
        
        print("\n4️⃣ パラメータ最適化")
        parameter_optimization_example()
        
        print("\n5️⃣ リアルタイムシミュレーション")
        realtime_simulation_example()
        
        print(f"\n🎉 全ての拡張機能例が完了しました！")
        print(f"💡 完全なデモは notebooks/TradingSystemPro_Demo.ipynb をご覧ください")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        print("💡 requirements.txt の依存関係を確認してください")
