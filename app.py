import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any
import os
from dotenv import load_dotenv
from agent import DataAnalysisAgent

# 環境変数を読み込み
load_dotenv()

def main():
    st.set_page_config(
        page_title="AI データ分析アシスタント",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 AI データ分析アシスタント")
    st.markdown("CSVファイルをアップロードするだけで、AIが自動的にデータ分析を行い、洞察レポートを生成します。")
    
    # サイドバーでAPIキーの設定
    with st.sidebar:
        st.header("🔧 設定")
        
        # OpenAI API キーの入力
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="OpenAI API キーを入力してください"
        )
        
        # モデル選択
        model_name = st.selectbox(
            "使用するモデル",
            ["gpt-4o", "gpt-4.1"],
            index=1
        )
        
        st.markdown("---")
        st.markdown("### 📝 使い方")
        st.markdown("""
        1. OpenAI API キーを入力
        2. CSVファイルをアップロード
        3. 「分析開始」ボタンをクリック
        4. AIが自動的に分析を実行
        5. 結果とレポートを確認
        """)
    
    # メインエリア
    if not api_key:
        st.warning("⚠️ サイドバーでOpenAI API キーを入力してください。")
        return
    
    # ファイルアップロード
    uploaded_file = st.file_uploader(
        "📁 CSVファイルをアップロードしてください",
        type=["csv"],
        help="分析したいCSVファイルを選択してください"
    )
    
    if uploaded_file is not None:
        try:
            # データフレームの読み込み
            df = pd.read_csv(uploaded_file)
            
            # データのプレビュー
            st.header("📋 データプレビュー")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("行数", df.shape[0])
            with col2:
                st.metric("列数", df.shape[1])
            with col3:
                st.metric("欠損値", df.isnull().sum().sum())
            
            # データの最初の数行を表示
            st.subheader("データの最初の10行")
            st.dataframe(df.head(10))
            
            # 分析開始ボタン
            if st.button("🚀 分析開始", type="primary"):
                analyze_data(df, api_key, model_name)
                
        except Exception as e:
            st.error(f"❌ ファイルの読み込みでエラーが発生しました: {str(e)}")

def analyze_data(df: pd.DataFrame, api_key: str, model_name: str):
    """データ分析を実行して結果を表示"""
    
    # プログレスバーとステータス
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # エージェントの初期化
        status_text.text("🤖 AIエージェントを初期化中...")
        progress_bar.progress(10)
        
        agent = DataAnalysisAgent(openai_api_key=api_key, model_name=model_name)
        
        # 分析実行
        status_text.text("🔍 データ分析を実行中...")
        progress_bar.progress(30)
        
        # セッション状態にストリーミング用のプレースホルダーを作成
        if 'analysis_container' not in st.session_state:
            st.session_state.analysis_container = st.container()
        
        with st.session_state.analysis_container:
            # リアルタイムでの進捗表示用のプレースホルダー
            progress_placeholder = st.empty()
            
            # 分析実行
            result = agent.analyze(df)
            
            progress_bar.progress(90)
            status_text.text("📝 結果を整理中...")
            
            if result["success"]:
                progress_bar.progress(100)
                status_text.text("✅ 分析完了!")
                
                # 結果の表示
                display_analysis_results(result)
                
            else:
                st.error(f"❌ 分析中にエラーが発生しました: {result.get('error', '不明なエラー')}")
                
    except Exception as e:
        st.error(f"❌ 分析の実行中にエラーが発生しました: {str(e)}")
    
    finally:
        progress_bar.empty()
        status_text.empty()

def display_analysis_results(result: Dict[str, Any]):
    """分析結果を表示"""
    
    st.header("📊 分析結果")
    
    # タブで結果を整理
    tab1, tab2, tab3 = st.tabs(["📝 レポート", "📈 グラフ", "🔍 詳細結果"])
    
    with tab1:
        st.subheader("🎯 AI生成レポート")
        if result.get("report"):
            st.markdown(result["report"])
        else:
            st.warning("レポートが生成されませんでした。")
    
    with tab2:
        st.subheader("📈 生成されたグラフ")
        
        # 実行結果からグラフを抽出して表示
        execution_results = result.get("execution_results", [])
        graph_count = 0
        
        # デバッグ情報
        st.write(f"実行結果の数: {len(execution_results)}")
        
        for i, exec_result in enumerate(execution_results):
            st.write(f"タスク {i+1}: {exec_result['task']['description']}")
            st.write(f"実行成功: {exec_result['result']['success']}")
            st.write(f"フィギュア数: {len(exec_result['result']['figures'])}")
            
            if exec_result["result"]["success"] and exec_result["result"]["figures"]:
                st.markdown(f"**{exec_result['task']['description']}**")
                
                for fig_info in exec_result["result"]["figures"]:
                    try:
                        st.plotly_chart(fig_info["figure"], use_container_width=True)
                        graph_count += 1
                    except Exception as e:
                        st.error(f"グラフの表示でエラーが発生しました: {str(e)}")
            elif exec_result["result"]["success"]:
                st.write("タスクは成功しましたが、フィギュアが検出されませんでした。")
                st.write(f"出力: {exec_result['result']['output']}")
        
        if graph_count == 0:
            st.info("表示可能なグラフがありませんでした。")
    
    with tab3:
        st.subheader("🔍 実行詳細")
        
        # データ概要
        if result.get("data_summary"):
            st.markdown("### 📋 データ概要")
            with st.expander("詳細を表示"):
                st.json(result["data_summary"])
        
        # 実行されたタスク
        st.markdown("### ⚡ 実行されたタスク")
        
        execution_results = result.get("execution_results", [])
        for i, exec_result in enumerate(execution_results, 1):
            with st.expander(f"タスク {i}: {exec_result['task']['description']}"):
                
                # タスクの詳細
                st.markdown("**タスク情報:**")
                st.json(exec_result["task"])
                
                # 生成されたコード
                st.markdown("**生成されたコード:**")
                st.code(exec_result["code"], language="python")
                
                # 実行結果
                st.markdown("**実行結果:**")
                if exec_result["result"]["success"]:
                    st.success("✅ 実行成功")
                    
                    if exec_result["result"]["output"]:
                        st.markdown("**出力:**")
                        st.text(exec_result["result"]["output"])
                    
                    if exec_result["result"]["figures"]:
                        st.markdown(f"**生成されたグラフ数:** {len(exec_result['result']['figures'])}")
                        
                else:
                    st.error("❌ 実行失敗")
                    st.markdown("**エラーメッセージ:**")
                    st.code(exec_result["result"]["error"], language="text")

def show_sample_data():
    """サンプルデータの表示"""
    st.header("📊 サンプルデータ")
    
    # サンプルデータの生成
    import numpy as np
    
    np.random.seed(42)
    sample_data = {
        "年齢": np.random.normal(35, 10, 100).astype(int),
        "年収": np.random.normal(500, 150, 100),
        "部署": np.random.choice(["営業", "開発", "マーケティング", "人事"], 100),
        "経験年数": np.random.poisson(5, 100),
        "満足度": np.random.choice([1, 2, 3, 4, 5], 100)
    }
    
    sample_df = pd.DataFrame(sample_data)
    
    st.dataframe(sample_df.head(10))
    
    # サンプルデータのダウンロード
    csv = sample_df.to_csv(index=False)
    st.download_button(
        label="📥 サンプルデータをダウンロード",
        data=csv,
        file_name="sample_data.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    # サイドバーにサンプルデータのセクションを追加
    with st.sidebar:
        st.markdown("---")
        if st.button("📊 サンプルデータを表示"):
            show_sample_data()
    
    main()
