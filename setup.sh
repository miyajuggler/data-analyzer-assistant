#!/bin/bash

echo "🚀 AI データ分析アシスタントのセットアップを開始します..."

# 仮想環境の作成
echo "📦 仮想環境を作成中..."
python -m venv venv

# 仮想環境のアクティベート
echo "⚡ 仮想環境をアクティベート中..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux/Mac
    source venv/bin/activate
fi

# パッケージのインストール
echo "📚 必要なパッケージをインストール中..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ セットアップが完了しました！"
echo ""
echo "🔧 次のステップ:"
echo "1. .env ファイルにOpenAI API キーを設定してください"
echo "2. 以下のコマンドでアプリケーションを起動してください:"
echo ""
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "   source venv/Scripts/activate"
else
    echo "   source venv/bin/activate"
fi
echo "   streamlit run app.py"
