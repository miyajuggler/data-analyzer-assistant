import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from typing import Dict, Any, List
import io
import sys
import builtins
from contextlib import redirect_stdout, redirect_stderr


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """データの概要情報を取得する"""
    # JSONシリアライズ可能な形式でデータ型を取得
    dtypes_dict = {}
    for col, dtype in df.dtypes.items():
        dtypes_dict[col] = str(dtype)
    
    # JSONシリアライズ可能な形式で欠損値数を取得
    null_counts_dict = {}
    for col, count in df.isnull().sum().items():
        null_counts_dict[col] = int(count)
    
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": dtypes_dict,
        "null_counts": null_counts_dict,
        "duplicate_count": int(df.duplicated().sum()),
        "memory_usage": int(df.memory_usage(deep=True).sum()),
    }
    
    # 数値カラムの基本統計量
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        numeric_stats = {}
        for col in numeric_cols:
            stats = df[col].describe()
            # pandas統計値をfloatに変換してJSONシリアライズ可能にする
            numeric_stats[col] = {
                'count': float(stats['count']),
                'mean': float(stats['mean']),
                'std': float(stats['std']),
                'min': float(stats['min']),
                '25%': float(stats['25%']),
                '50%': float(stats['50%']),
                '75%': float(stats['75%']),
                'max': float(stats['max'])
            }
        summary["numeric_stats"] = numeric_stats
    
    # カテゴリカルカラムの情報
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        summary["categorical_info"] = {}
        for col in categorical_cols:
            top_values = {}
            for value, count in df[col].value_counts().head(5).items():
                # 値とカウントをJSONシリアライズ可能な形式に変換
                top_values[str(value)] = int(count)
            
            summary["categorical_info"][col] = {
                "unique_count": int(df[col].nunique()),
                "top_values": top_values
            }
    
    return summary


def safe_code_execution(code: str, df: pd.DataFrame, task_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """安全にコードを実行し、結果を返す"""
    # 実行環境を制限
    allowed_imports = [
        'pandas', 'pd', 'numpy', 'np', 'plotly', 'px', 'go', 'ff',
        'matplotlib', 'plt', 'seaborn', 'sns', 'scipy', 'sklearn'
    ]
    
    # グローバル変数の設定
    import builtins
    safe_builtins = {}
    
    # 安全な組み込み関数のリスト
    safe_builtin_names = [
        'len', 'range', 'enumerate', 'zip', 'map', 'filter', 'sum',
        'min', 'max', 'abs', 'round', 'int', 'float', 'str', 'bool',
        'list', 'dict', 'tuple', 'set', 'print', 'type', 'isinstance',
        '__import__', 'getattr', 'hasattr', 'setattr', 'delattr',
        'dir', 'vars', 'sorted', 'reversed', 'any', 'all'
    ]
    
    for name in safe_builtin_names:
        if hasattr(builtins, name):
            safe_builtins[name] = getattr(builtins, name)
    
    exec_globals = {
        '__builtins__': safe_builtins,
        'pd': pd,
        'px': px,
        'go': go,
        'ff': ff,
        'np': np,
        'df': df.copy(),  # DataFrameのコピーを渡す
    }
    
    # 出力をキャプチャ
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    result = {
        "success": False,
        "output": "",
        "error": "",
        "figures": [],
        "variables": {}
    }
    
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, exec_globals)
        
        result["success"] = True
        result["output"] = stdout_capture.getvalue()
        
        # デバッグ情報：実行後の変数をチェック
        print(f"実行後の変数: {list(exec_globals.keys())}")
        
        # 作成されたfigureを収集
        figure_found = False
        for var_name, var_value in exec_globals.items():
            # より包括的なPlotlyオブジェクト検出
            var_type_str = str(type(var_value))
            is_plotly_figure = (
                'plotly' in var_type_str.lower() and 
                ('figure' in var_type_str.lower() or 'graph' in var_type_str.lower())
            ) or (
                hasattr(var_value, 'to_dict') and 
                hasattr(var_value, 'show') and
                hasattr(var_value, 'data')
            )
            
            if is_plotly_figure:
                result["figures"].append({
                    "name": var_name,
                    "figure": var_value
                })
                print(f"Plotlyフィギュアが見つかりました: {var_name} (型: {var_type_str})")
                figure_found = True
            elif var_name not in ['__builtins__', 'pd', 'px', 'go', 'ff', 'df', 'np']:
                # その他の変数も保存（デバッグ用）
                result["variables"][var_name] = str(var_value)[:1000]  # 長すぎる場合は切り詰め
        
        if not figure_found:
            # タスクタイプに基づいて適切なメッセージを表示
            task_type = task_info.get("task_type", "") if task_info else ""
            if task_type == "basic_info":
                print("基本情報表示タスクが完了しました")
            else:
                print("フィギュアが見つかりませんでした。利用可能な変数:")
                for var_name, var_value in exec_globals.items():
                    if var_name not in ['__builtins__', 'pd', 'px', 'go', 'ff', 'df', 'np']:
                        print(f"  {var_name}: {type(var_value)}")
                
    except Exception as e:
        result["error"] = str(e)
        result["output"] = stderr_capture.getvalue()
    
    return result


def generate_analysis_tasks(data_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """データ概要に基づいて分析タスクを生成する（厳密に5個に制限）"""
    tasks = []
    
    # タスク1: 基本的な情報表示（必須）
    tasks.append({
        "task_type": "basic_info",
        "description": "データの基本情報を表示"
    })
    
    # 数値カラムがある場合のタスク
    if "numeric_stats" in data_summary:
        numeric_cols = list(data_summary["numeric_stats"].keys())
        
        # タスク2: 最初の数値カラムのヒストグラム
        if len(numeric_cols) > 0 and len(tasks) < 5:
            tasks.append({
                "task_type": "histogram",
                "column": numeric_cols[0],
                "description": f"{numeric_cols[0]}のヒストグラム"
            })
        
        # タスク3: 2番目の数値カラムのヒストグラム（存在する場合）
        if len(numeric_cols) > 1 and len(tasks) < 5:
            tasks.append({
                "task_type": "histogram",
                "column": numeric_cols[1],
                "description": f"{numeric_cols[1]}のヒストグラム"
            })
        
        # タスク4: 数値カラムが3つ以上ある場合は3番目のヒストグラム
        if len(numeric_cols) > 2 and len(tasks) < 5:
            tasks.append({
                "task_type": "histogram",
                "column": numeric_cols[2],
                "description": f"{numeric_cols[2]}のヒストグラム"
            })
        
        # タスク5: 相関行列（数値カラムが2つ以上の場合）
        if len(numeric_cols) > 1 and len(tasks) < 5:
            tasks.append({
                "task_type": "correlation_matrix",
                "description": "数値カラム間の相関行列"
            })
    
    # 数値カラムが少ない場合、カテゴリカルカラムで埋める
    if "categorical_info" in data_summary and len(tasks) < 5:
        categorical_cols = list(data_summary["categorical_info"].keys())
        for col in categorical_cols:
            if len(tasks) < 5:
                tasks.append({
                    "task_type": "bar_chart",
                    "column": col,
                    "description": f"{col}の分布（棒グラフ）"
                })
    
    # 厳密に5個に制限
    tasks = tasks[:5]
    
    # 不足している場合は基本情報で埋める
    while len(tasks) < 5:
        tasks.append({
            "task_type": "basic_info",
            "description": f"データの基本情報を表示（タスク{len(tasks)+1}）"
        })
    
    return tasks


def create_analysis_code(task: Dict[str, Any]) -> str:
    """分析タスクに基づいてPythonコードを生成する"""
    task_type = task["task_type"]
    
    if task_type == "basic_info":
        return """
# データの基本情報
print("=== データの基本情報 ===")
print(f"データサイズ: {df.shape}")
print(f"カラム: {list(df.columns)}")
print("\\n=== データ型 ===")
print(df.dtypes)
print("\\n=== 欠損値の数 ===")
print(df.isnull().sum())
print("\\n=== 基本統計量 ===")
print(df.describe())

# データ概要の可視化
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 欠損値の可視化
null_counts = df.isnull().sum()
if null_counts.sum() > 0:
    # 欠損値がある場合
    fig = go.Figure(data=[
        go.Bar(x=null_counts.index, y=null_counts.values)
    ])
    fig.update_layout(
        title="各カラムの欠損値数",
        xaxis_title="カラム",
        yaxis_title="欠損値数"
    )
else:
    # 欠損値がない場合はデータ型の分布を表示
    dtype_counts = df.dtypes.value_counts()
    fig = go.Figure(data=[
        go.Pie(labels=dtype_counts.index.astype(str), values=dtype_counts.values)
    ])
    fig.update_layout(title="データ型の分布")

print("データ概要の可視化を作成しました")
"""
    
    elif task_type == "histogram":
        col = task["column"]
        return f"""
# {col}のヒストグラム
import plotly.express as px
fig = px.histogram(df, x='{col}', title=f'{col}の分布')
print(f"ヒストグラムを作成しました: {col}")
# figを変数として保持（show()は呼ばない）
"""
    
    elif task_type == "correlation_matrix":
        return """
# 相関行列
import plotly.express as px
numeric_cols = df.select_dtypes(include=['number']).columns
if len(numeric_cols) > 1:
    corr_matrix = df[numeric_cols].corr()
    fig = px.imshow(corr_matrix, 
                    text_auto=True, 
                    aspect="auto",
                    title="数値カラム間の相関行列")
    print("相関行列を作成しました")
    # figを変数として保持（show()は呼ばない）
else:
    print("相関行列を作成するには2つ以上の数値カラムが必要です")
"""
    
    elif task_type == "scatter_matrix":
        columns = task.get("columns", [])
        cols_str = str(columns)
        return f"""
# 散布図行列
import plotly.express as px
fig = px.scatter_matrix(df, dimensions={cols_str}, title="散布図行列")
# figを変数として保持（show()は呼ばない）
"""
    
    elif task_type == "bar_chart":
        col = task["column"]
        return f"""
# {col}の分布（棒グラフ）
import plotly.express as px
value_counts = df['{col}'].value_counts().head(10)
fig = px.bar(x=value_counts.index, y=value_counts.values, 
             title=f'{col}の分布 (上位10件)',
             labels={{'x': '{col}', 'y': '件数'}})
print(f"棒グラフを作成しました: {col}")
# figを変数として保持（show()は呼ばない）
"""
    
    else:
        return f"# 未対応のタスクタイプ: {task_type}"
