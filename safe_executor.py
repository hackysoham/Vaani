"""
Safe Executor for VaaniData
Safely executes Pandas expressions against user DataFrames and generates data takeaways.
"""

import ast
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple


# Disallowed AST nodes and attributes for safety
DISALLOWED_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Delete,
    ast.Global,
    ast.Nonlocal,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Raise,
    ast.Assert,
)

DISALLOWED_CALLS = {
    "eval", "exec", "compile", "__import__", "open", "input",
    "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr",
    "exit", "quit", "help", "breakpoint"
}

DISALLOWED_ATTRS = {
    "__class__", "__base__", "__bases__", "__subclasses__", "__mro__",
    "__globals__", "__code__", "__closure__", "__builtins__",
    "to_pickle", "to_sql", "to_parquet", "to_feather", "to_hdf"
}


def is_safe_expression(code_str: str) -> Tuple[bool, str]:
    """
    Statically analyzes python code string to ensure it's a read-only Pandas expression.
    """
    clean_code = code_str.strip()
    if not clean_code:
        return False, "Empty code expression."
    
    # Strip markdown code fences if LLM wrapped it in ```python ... ```
    if clean_code.startswith("```"):
        lines = clean_code.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        clean_code = "\n".join(lines).strip()

    try:
        tree = ast.parse(clean_code, mode="eval")
    except SyntaxError:
        try:
            tree = ast.parse(clean_code, mode="exec")
        except SyntaxError as e:
            return False, f"Syntax error in code: {e}"

    for node in ast.walk(tree):
        if isinstance(node, DISALLOWED_NODES):
            return False, f"Disallowed statement type: {type(node).__name__}"
        
        # Check function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in DISALLOWED_CALLS:
                return False, f"Disallowed function call: {node.func.id}()"
            elif isinstance(node.func, ast.Attribute) and node.func.attr in DISALLOWED_ATTRS:
                return False, f"Disallowed attribute access: {node.func.attr}"

        # Check attribute access
        if isinstance(node, ast.Attribute) and node.attr in DISALLOWED_ATTRS:
            return False, f"Disallowed attribute access: {node.attr}"

    return True, clean_code


def compute_takeaway(result_df: pd.DataFrame, code_str: str) -> str:
    """
    Computes a concrete 1-sentence data takeaway from the top row or metric of result_df.
    """
    if result_df is None or result_df.empty:
        return "No matching records found for the specified criteria."

    try:
        rows, cols = result_df.shape
        col_names = [str(c) for c in result_df.columns]

        # Case 1: Single cell or 1x1 summary (scalar or single aggregation)
        if rows == 1 and cols == 1:
            val = result_df.iloc[0, 0]
            col = col_names[0]
            if isinstance(val, (int, float, np.number)):
                formatted_val = f"₹{val:,.2f}" if "revenue" in col.lower() or "profit" in col.lower() or "price" in col.lower() else f"{val:,.2f}"
                return f"Key Metric takeaway: The computed {col.replace('_', ' ')} is {formatted_val}."
            return f"Key takeaway: The result is {val}."

        # Case 2: 1-row multi-column summary
        if rows == 1:
            items = []
            for c in col_names[:3]:
                v = result_df.iloc[0][c]
                if isinstance(v, (int, float, np.number)):
                    formatted = f"₹{v:,.2f}" if any(k in c.lower() for k in ["revenue", "profit", "price", "amount"]) else f"{v:,.2f}"
                    items.append(f"{c.replace('_', ' ')} = {formatted}")
                else:
                    items.append(f"{c.replace('_', ' ')}: {v}")
            return f"Key takeaway: Summary highlights - {', '.join(items)}."

        # Case 3: Ranked / Aggregated table (e.g. Grouped by city or category)
        top_row = result_df.iloc[0]
        # Find numeric column if any
        num_cols = [c for c in result_df.columns if pd.api.types.is_numeric_dtype(result_df[c])]
        cat_cols = [c for c in result_df.columns if not pd.api.types.is_numeric_dtype(result_df[c])]

        if cat_cols and num_cols:
            main_cat_col = cat_cols[0]
            main_num_col = num_cols[0]
            top_entity = top_row[main_cat_col]
            top_val = top_row[main_num_col]
            
            # Check if there is total sum to calculate share
            total_val = result_df[main_num_col].sum()
            share_text = ""
            if total_val > 0 and isinstance(top_val, (int, float, np.number)):
                share_pct = (top_val / total_val) * 100
                share_text = f" (accounting for {share_pct:.1f}% share of this cohort)"

            val_str = f"₹{top_val:,.2f}" if any(k in main_num_col.lower() for k in ["revenue", "profit", "amount", "price"]) else f"{top_val:,.2f}"
            return f"Leading takeaway: {top_entity} ranks #1 in {main_num_col.replace('_', ' ')} with {val_str}{share_text}."

        # Fallback
        first_col = col_names[0]
        first_val = top_row[first_col]
        return f"Top result: Ranked first is '{first_val}' across {rows} evaluated entries."
    except Exception as e:
        return f"Evaluated {len(result_df)} resulting records successfully."


def execute_pandas_query(df: pd.DataFrame, code_str: str) -> Dict[str, Any]:
    """
    Executes pandas_code safely in an isolated environment against df.
    Returns:
    {
        "success": bool,
        "clean_code": str,
        "result_df": pd.DataFrame or None,
        "takeaway": str,
        "error": str or None
    }
    """
    is_safe, processed_code = is_safe_expression(code_str)
    if not is_safe:
        return {
            "success": False,
            "clean_code": code_str,
            "result_df": None,
            "takeaway": "",
            "error": f"Security validation failed: {processed_code}"
        }

    # Prepare execution sandbox
    sandbox_globals = {
        "__builtins__": {
            "min": min, "max": max, "sum": sum, "len": len, "round": round,
            "abs": abs, "range": range, "list": list, "dict": dict, "set": set,
            "str": str, "int": int, "float": float, "bool": bool
        },
        "pd": pd,
        "np": np
    }
    sandbox_locals = {"df": df.copy()}

    try:
        # Evaluate expression
        output = eval(processed_code, sandbox_globals, sandbox_locals)

        # Standardize result into a presentable DataFrame
        if isinstance(output, pd.DataFrame):
            result_df = output
        elif isinstance(output, pd.Series):
            result_df = output.reset_index()
            # If column names are 0 or index, rename cleanly
            if len(result_df.columns) == 2 and result_df.columns[1] == 0:
                result_df.columns = [result_df.columns[0], "value"]
        elif isinstance(output, (int, float, str, bool, np.number)):
            result_df = pd.DataFrame([{"result": output}])
        elif isinstance(output, (list, tuple)):
            result_df = pd.DataFrame(output, columns=["value"])
        elif isinstance(output, dict):
            result_df = pd.DataFrame(list(output.items()), columns=["metric", "value"])
        else:
            result_df = pd.DataFrame([{"result": str(output)}])

        # Generate takeaway
        takeaway = compute_takeaway(result_df, processed_code)

        return {
            "success": True,
            "clean_code": processed_code,
            "result_df": result_df,
            "takeaway": takeaway,
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "clean_code": processed_code,
            "result_df": None,
            "takeaway": "",
            "error": f"Pandas execution error: {type(e).__name__}: {str(e)}"
        }
