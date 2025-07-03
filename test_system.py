import pandas as pd
import numpy as np

def create_sample_data():
    """サンプルデータを作成"""
    np.random.seed(42)
    
    # 顧客データのサンプル
    n_customers = 1000
    
    data = {
        "customer_id": range(1, n_customers + 1),
        "age": np.random.normal(35, 12, n_customers).astype(int),
        "annual_income": np.random.normal(50000, 15000, n_customers),
        "spending_score": np.random.normal(50, 25, n_customers),
        "gender": np.random.choice(["Male", "Female"], n_customers),
        "city": np.random.choice(["Tokyo", "Osaka", "Kyoto", "Yokohama", "Kobe"], n_customers),
        "membership_years": np.random.poisson(3, n_customers),
        "purchase_frequency": np.random.poisson(12, n_customers),
        "satisfaction_score": np.random.choice([1, 2, 3, 4, 5], n_customers, p=[0.05, 0.15, 0.3, 0.35, 0.15])
    }
    
    # データの調整
    data["age"] = np.clip(data["age"], 18, 80)
    data["annual_income"] = np.clip(data["annual_income"], 20000, 150000)
    data["spending_score"] = np.clip(data["spending_score"], 0, 100)
    data["membership_years"] = np.clip(data["membership_years"], 0, 20)
    data["purchase_frequency"] = np.clip(data["purchase_frequency"], 0, 50)
    
    df = pd.DataFrame(data)
    
    # 欠損値をいくつか作成（リアルなデータシミュレーション）
    missing_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
    df.loc[missing_indices, "annual_income"] = np.nan
    
    return df

def test_tools():
    """tools.pyの機能をテスト"""
    print("🧪 tools.pyの機能をテスト中...")
    
    try:
        from tools import get_data_summary, generate_analysis_tasks, create_analysis_code
        
        # サンプルデータの作成
        df = create_sample_data()
        print(f"✅ サンプルデータ作成完了: {df.shape}")
        
        # データ概要の取得
        summary = get_data_summary(df)
        print(f"✅ データ概要取得完了: {len(summary)} 項目")
        
        # 分析タスクの生成
        tasks = generate_analysis_tasks(summary)
        print(f"✅ 分析タスク生成完了: {len(tasks)} タスク")
        
        # コード生成のテスト
        if tasks:
            code = create_analysis_code(tasks[0])
            print(f"✅ コード生成完了: {len(code)} 文字")
        
        print("🎉 tools.pyのテストが完了しました！")
        return True
        
    except Exception as e:
        print(f"❌ tools.pyのテストでエラーが発生: {e}")
        return False

def test_basic_imports():
    """基本的なインポートをテスト"""
    print("📦 基本的なインポートをテスト中...")
    
    try:
        import streamlit as st
        print("✅ Streamlit インポート成功")
        
        import pandas as pd
        print("✅ Pandas インポート成功")
        
        import plotly.express as px
        print("✅ Plotly インポート成功")
        
        import openai
        print("✅ OpenAI インポート成功")
        
        from dotenv import load_dotenv
        print("✅ python-dotenv インポート成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        return False

def main():
    """メイン関数"""
    print("🚀 システムテストを開始します...")
    print("=" * 50)
    
    # 基本的なインポートテスト
    if not test_basic_imports():
        print("❌ 基本的なインポートテストに失敗しました")
        return
    
    print("=" * 50)
    
    # ツールのテスト
    if not test_tools():
        print("❌ ツールテストに失敗しました")
        return
    
    print("=" * 50)
    
    # サンプルデータの保存
    try:
        df = create_sample_data()
        df.to_csv("sample_customer_data.csv", index=False)
        print("✅ サンプルデータをsample_customer_data.csvに保存しました")
    except Exception as e:
        print(f"❌ サンプルデータの保存に失敗: {e}")
    
    print("=" * 50)
    print("🎉 システムテストが完了しました！")
    print("📋 次のステップ:")
    print("1. .envファイルにOpenAI API キーを設定")
    print("2. 'streamlit run app.py'でアプリケーションを起動")
    print("3. sample_customer_data.csvを使用してテスト")

if __name__ == "__main__":
    main()
