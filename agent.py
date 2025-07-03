from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import pandas as pd
import json
from tools import (
    get_data_summary, 
    safe_code_execution, 
    generate_analysis_tasks, 
    create_analysis_code
)


def safe_json_dumps(obj, **kwargs):
    """JSONシリアライズ可能でない型を処理するカスタム関数"""
    def default_converter(o):
        if hasattr(o, 'dtype'):
            if 'int' in str(o.dtype):
                return int(o)
            elif 'float' in str(o.dtype):
                return float(o)
            else:
                return str(o)
        elif hasattr(o, 'tolist'):
            return o.tolist()
        return str(o)
    
    return json.dumps(obj, default=default_converter, **kwargs)


class AnalysisState(TypedDict):
    """グラフ全体で共有する状態"""
    dataframe: Optional[pd.DataFrame]
    data_summary: Optional[Dict[str, Any]]
    plan: Optional[List[Dict[str, Any]]]
    current_task_index: int
    code_string: Optional[str]
    execution_results: List[Dict[str, Any]]
    report: Optional[str]
    error_count: int
    max_retries: int
    completed: bool


class DataAnalysisAgent:
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4.1"):
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=model_name,
            temperature=0.1
        )
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """LangGraphのステートグラフを作成"""
        workflow = StateGraph(AnalysisState)
        
        # ノードを追加
        workflow.add_node("situation_awareness", self.situation_awareness_node)
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("coder", self.coder_node)
        workflow.add_node("code_execution", self.code_execution_node)
        workflow.add_node("code_revision", self.code_revision_node)
        workflow.add_node("task_manager", self.task_manager_node)
        workflow.add_node("reporter", self.reporter_node)
        workflow.add_node("reviewer", self.reviewer_node)
        
        # エントリーポイントの設定
        workflow.set_entry_point("situation_awareness")
        
        # エッジ（条件分岐）を追加
        workflow.add_edge("situation_awareness", "planner")
        workflow.add_edge("planner", "coder")
        workflow.add_edge("coder", "code_execution")
        
        # 条件分岐: コード実行後の処理
        workflow.add_conditional_edges(
            "code_execution",
            self.decide_after_execution,
            {
                "retry": "code_revision",
                "next_task": "task_manager",
                "error": END
            }
        )
        
        workflow.add_edge("code_revision", "code_execution")
        workflow.add_conditional_edges(
            "task_manager",
            self.decide_next_action,
            {
                "continue": "coder",
                "complete": "reporter"
            }
        )
        workflow.add_edge("reporter", "reviewer")
        workflow.add_edge("reviewer", END)
        
        # 再帰制限を増加させる設定を追加
        return workflow.compile()
    
    def situation_awareness_node(self, state: AnalysisState) -> AnalysisState:
        """状況把握ノード: データの前処理と傾向把握"""
        print("🔍 データの状況を把握中...")
        
        if state["dataframe"] is None:
            raise ValueError("DataFrameが提供されていません")
        
        # データの概要を取得
        data_summary = get_data_summary(state["dataframe"])
        
        state["data_summary"] = data_summary
        return state
    
    def planner_node(self, state: AnalysisState) -> AnalysisState:
        """プランナーノード: 分析計画を立案"""
        print("📋 分析計画を立案中...")
        
        # 基本的な分析タスクを生成（10個に制限）
        basic_tasks = generate_analysis_tasks(state["data_summary"])
        
        # 基本タスクのみを使用し、10個に制限
        state["plan"] = basic_tasks[:10]
        print(f"📋 分析計画を確定しました: {len(state['plan'])}個のタスク")
        
        # タスクの内容を表示
        for i, task in enumerate(state["plan"], 1):
            print(f"  タスク{i}: {task['description']}")
        
        state["current_task_index"] = 0
        state["execution_results"] = []
        state["error_count"] = 0
        state["max_retries"] = 3
        
        return state
    
    def coder_node(self, state: AnalysisState) -> AnalysisState:
        """コーダーノード: Python コードを生成"""
        current_index = state["current_task_index"]
        
        if current_index >= len(state["plan"]):
            state["completed"] = True
            return state
        
        current_task = state["plan"][current_index]
        print(f"💻 コード生成中: {current_task['description']}")
        
        # 基本的なコード生成
        basic_code = create_analysis_code(current_task)
        
        # LLMを使ってコードを改良
        system_prompt = """
あなたは経験豊富なPythonデータサイエンティストです。
提供された分析タスクに対して、効果的で読みやすいPythonコードを生成してください。

利用可能なライブラリ:
- pandas (df変数でDataFrameが利用可能)
- plotly.express as px
- plotly.graph_objects as go
- plotly.figure_factory as ff

注意事項:
- グラフを作成する場合、figオブジェクトを変数として保持してください
- fig.show()は呼び出さないでください（システムが自動で表示します）
- エラーハンドリングを含めてください
- コメントを適切に追加してください
- 変数名は分かりやすくしてください
"""
        
        user_prompt = f"""
分析タスク:
{safe_json_dumps(current_task, ensure_ascii=False, indent=2)}

データの概要:
{safe_json_dumps(state['data_summary'], ensure_ascii=False, indent=2)}

基本コード:
{basic_code}

この基本コードを改良して、より効果的で洞察に富んだ分析コードを生成してください。
"""
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            # コードブロックを抽出
            content = response.content
            if "```python" in content:
                code_start = content.find("```python") + 9
                code_end = content.find("```", code_start)
                enhanced_code = content[code_start:code_end].strip()
            elif "```" in content:
                code_start = content.find("```") + 3
                code_end = content.find("```", code_start)
                enhanced_code = content[code_start:code_end].strip()
            else:
                enhanced_code = basic_code
            
            state["code_string"] = enhanced_code
            
        except Exception as e:
            print(f"LLMによるコード生成でエラーが発生: {e}")
            state["code_string"] = basic_code
        
        return state
    
    def code_execution_node(self, state: AnalysisState) -> AnalysisState:
        """コード実行ノード: 生成されたコードを実行"""
        print("⚡ コードを実行中...")
        
        if not state["code_string"]:
            return state
        
        # 現在のタスク情報を取得
        current_task = state["plan"][state["current_task_index"]]
        
        # コードを安全に実行（タスク情報を渡す）
        execution_result = safe_code_execution(state["code_string"], state["dataframe"], current_task)
        
        # 結果を状態に追加
        result_with_task = {
            "task": current_task,
            "code": state["code_string"],
            "result": execution_result
        }
        
        state["execution_results"].append(result_with_task)
        
        # エラーカウントの更新
        if not execution_result["success"]:
            state["error_count"] += 1
        else:
            state["error_count"] = 0  # 成功時はリセット
        
        return state
    
    def code_revision_node(self, state: AnalysisState) -> AnalysisState:
        """コード修正ノード: エラー時にコードを修正"""
        print("🔧 コードを修正中...")
        
        last_result = state["execution_results"][-1]
        error_message = last_result["result"]["error"]
        
        system_prompt = """
あなたは経験豊富なPythonデバッガーです。
実行時エラーが発生したコードを修正してください。

利用可能なライブラリ:
- pandas (df変数でDataFrameが利用可能)
- plotly.express as px
- plotly.graph_objects as go
- plotly.figure_factory as ff

注意事項:
- グラフを作成する場合、figオブジェクトを変数として保持してください
- fig.show()は呼び出さないでください（システムが自動で表示します）
- 修正されたコードのみを返してください
"""
        
        user_prompt = f"""
エラーが発生したコード:
```python
{state['code_string']}
```

エラーメッセージ:
{error_message}

データの概要:
{safe_json_dumps(state['data_summary'], ensure_ascii=False, indent=2)}

このエラーを修正したコードを生成してください。
"""
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            # 修正されたコードを抽出
            content = response.content
            if "```python" in content:
                code_start = content.find("```python") + 9
                code_end = content.find("```", code_start)
                revised_code = content[code_start:code_end].strip()
            elif "```" in content:
                code_start = content.find("```") + 3
                code_end = content.find("```", code_start)
                revised_code = content[code_start:code_end].strip()
            else:
                revised_code = content.strip()
            
            state["code_string"] = revised_code
            
        except Exception as e:
            print(f"コード修正でエラーが発生: {e}")
        
        return state
    
    def reporter_node(self, state: AnalysisState) -> AnalysisState:
        """報告ノード: 分析結果を基にレポートを作成"""
        print("📝 レポートを作成中...")
        
        # 実行結果をまとめる
        results_summary = []
        for result in state["execution_results"]:
            if result["result"]["success"]:
                summary = {
                    "task": result["task"]["description"],
                    "output": result["result"]["output"],
                    "figures_count": len(result["result"]["figures"])
                }
                results_summary.append(summary)
        
        system_prompt = """
あなたは経験豊富なデータアナリストです。
実行された分析結果に基づいて、包括的で洞察に富んだレポートを作成してください。

レポートは以下の構成で作成してください：
1. エグゼクティブサマリー
2. データの概要
3. 主要な発見
4. 詳細な分析結果
5. 結論と推奨事項

レポートはマークダウン形式で、ビジネス関係者にも理解しやすい言葉で書いてください。
"""
        
        user_prompt = f"""
データの概要:
{safe_json_dumps(state['data_summary'], ensure_ascii=False, indent=2)}

実行された分析:
{safe_json_dumps(results_summary, ensure_ascii=False, indent=2)}

この分析結果に基づいて、包括的なデータ分析レポートを作成してください。
"""
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            state["report"] = response.content
            
        except Exception as e:
            print(f"レポート生成でエラーが発生: {e}")
            state["report"] = "レポートの生成中にエラーが発生しました。"
        
        return state
    
    def reviewer_node(self, state: AnalysisState) -> AnalysisState:
        """レビューノード: レポートの最終チェック"""
        print("🔍 レポートの最終チェック中...")
        
        system_prompt = """
あなたは経験豊富な編集者です。
提供されたデータ分析レポートをレビューし、以下の点をチェックして改善してください：

1. 文章の明確性と読みやすさ
2. 論理的な構成
3. 専門用語の適切な説明
4. 結論の妥当性
5. 誤字脱字や表現の修正

改善されたレポートを返してください。
"""
        
        user_prompt = f"""
レビュー対象のレポート:
{state['report']}

このレポートをレビューし、改善版を返してください。
"""
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            state["report"] = response.content
            
        except Exception as e:
            print(f"レポートレビューでエラーが発生: {e}")
        
        return state
    
    def decide_after_execution(self, state: AnalysisState) -> str:
        """コード実行後の条件分岐を決定"""
        if not state["execution_results"]:
            return "error"
            
        last_result = state["execution_results"][-1]
        
        # エラーが発生した場合
        if not last_result["result"]["success"]:
            if state["error_count"] >= state["max_retries"]:
                print(f"最大試行回数({state['max_retries']})に達しました。タスクをスキップします。")
                # エラーが続く場合は次のタスクへ進む - 状態更新は別のノードで行う
                return "next_task"
            else:
                return "retry"
        
        # 成功した場合、次のタスクへ - 状態更新は別のノードで行う
        return "next_task"
    
    def task_manager_node(self, state: AnalysisState) -> AnalysisState:
        """タスク管理ノード: 次のタスクへの進行を管理"""
        print("📋 タスクの進行を管理中...")
        
        # 成功した場合またはエラーが最大回数に達した場合、次のタスクへ
        if state["execution_results"]:
            last_result = state["execution_results"][-1]
            if last_result["result"]["success"] or state["error_count"] >= state["max_retries"]:
                state["current_task_index"] += 1
                state["error_count"] = 0  # エラーカウントをリセット
        
        return state
    
    def decide_next_action(self, state: AnalysisState) -> str:
        """次のアクションを決定"""
        if state["current_task_index"] >= len(state["plan"]):
            return "complete"
        else:
            return "continue"

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """データ分析を実行"""
        initial_state = AnalysisState(
            dataframe=df,
            data_summary=None,
            plan=None,
            current_task_index=0,
            code_string=None,
            execution_results=[],
            report=None,
            error_count=0,
            max_retries=3,
            completed=False
        )
        
        print("🚀 データ分析を開始します...")
        print(f"📊 データサイズ: {df.shape}")
        
        try:
            # 設定に再帰制限を追加して実行
            config = {"recursion_limit": 100}
            print(f"🔧 再帰制限: {config['recursion_limit']}")
            final_state = self.graph.invoke(initial_state, config)
            
            print("✅ 分析が正常に完了しました")
            return {
                "success": True,
                "report": final_state.get("report", ""),
                "execution_results": final_state.get("execution_results", []),
                "data_summary": final_state.get("data_summary", {})
            }
            
        except Exception as e:
            print(f"❌ 分析中にエラーが発生しました: {e}")
            import traceback
            print(f"詳細なエラー情報:\n{traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "report": "",
                "execution_results": [],
                "data_summary": {}
            }
