# ゴール

CSV をアップロードするだけで、AI が自律的にデータ分析を行い、グラフ付きの洞察レポートを生成する Streamlit アプリケーションを開発する。

# あなたへの指示

この設計に基づき、Python の LangGraph と OpenAI API を使って、データ分析 AI エージェントのバックエンド処理の基本構造を実装してください。
まずは、各エージェントに対応するノードと、それらを条件に応じてつなぐグラフ（StatefulGraph）の骨組みを作成してください。

# 全体構成

- **UI**: Streamlit
- **AI オーケストレーション**: LangGraph
- **LLM**: OpenAI API (gpt-4o など)
- **データ処理**: Pandas
- **グラフ描画**: Plotly

# 想定するファイル構成

- `app.py`: Streamlit の UI 部分
- `agent.py`: LangGraph を使ったエージェント全体の定義
- `tools.py`: データ分析やコード実行に使われる具体的な関数群

# エージェントの設計（LangGraph のノードとして実装）

## 状態管理 (State)

グラフ全体で共有する状態を定義する。

- `dataframe`: 分析対象の DataFrame
- `data_summary`: データの概要テキスト
- `plan`: 分析計画のリスト（JSON）
- `code_string`: 生成された Python コード
- `execution_results`: コードの実行結果（テキストや Plotly の figure オブジェクト）
- `report`: 生成されたレポート（マークダウン形式）
- `error_count`: コード修正の試行回数

## 1. 状況把握ノード (Situation Awareness Node)

- **役割**: データの前処理と傾向把握。
- **Input**: `dataframe`
- **Output**: `data_summary`（基本統計量、欠損値情報、カラム型などを格納）

## 2. プランナーノード (Planner Node)

- **役割**: データ概要に基づき、分析計画を立案する。
- **Input**: `data_summary`
- **Output**: `plan`（実行すべき分析タスクの JSON リスト。例: `[{"task_type": "histogram", "column": "age"}, {"task_type": "correlation_matrix"}]`）

## 3. コーダーノード (Coder Node)

- **役割**: 計画に基づき、データ分析の Python コードを生成する。
- **Input**: `plan`の各タスク
- **Output**: `code_string`

## 4. コード実行ノード (Code Execution Node)

- **役割**: 生成されたコードを実行する。`exec()`関数などを使用するが、セキュリティには注意が必要。
- **Input**: `code_string`
- **Output**: `execution_results`。エラーが発生した場合はエラー情報を返す。

## 5. コード修正ノード (Code Revision Node)

- **役割**: 実行時エラーが発生した場合にコードを修正する。
- **Input**: エラーになった `code_string` とエラーメッセージ
- **Output**: 修正された `code_string`

## 6. 報告ノード (Reporter Node)

- **役割**: 全ての分析結果を基に、グラフの解釈やインサイトを含むレポートを作成する。
- **Input**: `execution_results`
- **Output**: `report` (マークダウン形式)

## 7. レビューノード (Reviewer Node)

- **役割**: 生成されたレポートを最終チェックし、不適切な表現などを修正する。
- **Input**: `report`
- **Output**: 最終版の `report`

# グラフの条件分岐 (Conditional Edges)

- コード実行後、エラーがあれば「コード修正ノード」へ。成功すれば次のタスクへ。
- コード修正が規定回数（例: 3 回）失敗したら処理を中断。
- 全ての計画が完了したら「報告ノード」へ。
