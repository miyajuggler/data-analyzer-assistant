@echo off
echo 🚀 AI データ分析アシスタントのセットアップを開始します...

REM 仮想環境の作成
echo 📦 仮想環境を作成中...
python -m venv venv

REM 仮想環境のアクティベート
echo ⚡ 仮想環境をアクティベート中...
call venv\Scripts\activate.bat

REM パッケージのインストール
echo 📚 必要なパッケージをインストール中...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo ✅ セットアップが完了しました！
echo.
echo 🔧 次のステップ:
echo 1. .env ファイルにOpenAI API キーを設定してください
echo 2. 以下のコマンドでアプリケーションを起動してください:
echo.
echo    venv\Scripts\activate.bat
echo    streamlit run app.py

pause
