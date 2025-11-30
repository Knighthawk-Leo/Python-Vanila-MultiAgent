"""
Code Interpreter Agent - Executes Python code for data analysis
"""

import google.generativeai as genai
from typing import Dict, Any, List
import io
import sys
import traceback
import pandas as pd
import numpy as np
from contextlib import redirect_stdout, redirect_stderr
import ast
import os

from .base_agent import BaseAgent, AgentResult, Message


class CodeInterpreterAgent(BaseAgent):
    

    def __init__(self, api_key: str):
        super().__init__(name="CodeInterpreter", api_key=api_key)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.execution_results = []

    def get_capabilities(self) -> List[str]:
        return [
            "Execute Python code for data analysis",
            "Load and analyze CSV files",
            "Perform statistical analysis",
            "Data cleaning and preprocessing",
            "Generate insights from data",
        ]

    async def process(self, input_data: Dict[str, Any]) -> AgentResult:
        
        query = input_data.get("query", "")
        context = input_data.get("context", {})
        files = input_data.get("files", {})

        if files:
            for filename, filepath in files.items():
                try:
                    df = pd.read_csv(filepath)
                    self.dataframes[filename] = df
                    context[f"df_{filename}"] = {
                        "shape": df.shape,
                        "columns": df.columns.tolist(),
                        "dtypes": df.dtypes.to_dict(),
                        "head": df.head().to_dict(),
                        "summary": df.describe().to_dict() if len(df) > 0 else {},
                    }
                except Exception as e:
                    return AgentResult(
                        success=False,
                        data=None,
                        message=f"Error loading CSV file {filename}: {str(e)}",
                        agent_name=self.name,
                    )

        prompt = self._build_prompt(query, context)

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text

            code_blocks = self._extract_code_blocks(response_text)

            if not code_blocks:
                return AgentResult(
                    success=True,
                    data={
                        "analysis": response_text,
                        "code_executed": None,
                        "results": None,
                        "dataframes_info": {
                            k: v.shape for k, v in self.dataframes.items()
                        },
                    },
                    message="Analysis completed without code execution",
                    agent_name=self.name,
                    next_agent=None,
                )

            execution_results = []
            for i, code in enumerate(code_blocks):
                result = self._execute_code(code)
                execution_results.append(result)

            needs_visualization = self._needs_visualization(query, execution_results)

            return AgentResult(
                success=True,
                data={
                    "analysis": response_text,
                    "code_executed": code_blocks,
                    "results": execution_results,
                    "dataframes": {k: v.to_dict() for k, v in self.dataframes.items()},
                    "dataframes_info": {
                        k: {"shape": v.shape, "columns": v.columns.tolist()}
                        for k, v in self.dataframes.items()
                    },
                },
                message="Code execution completed successfully",
                agent_name=self.name,
                next_agent="VisualizationAgent" if needs_visualization else "AnswerSynthesiser",
                metadata={"needs_visualization": needs_visualization},
            )

        except Exception as e:
            return AgentResult(
                success=False,
                data={"error": str(e), "traceback": traceback.format_exc()},
                message=f"Error during code interpretation: {str(e)}",
                agent_name=self.name,
            )

    def _build_prompt(self, query: str, context: Dict[str, Any]) -> str:
        prompt = f"""You are a Python data analysis expert. Analyze the user's query and provide Python code to answer it.

User Query: {query}

"""
        if self.dataframes:
            prompt += "Available DataFrames:\n"
            for name, df in self.dataframes.items():
                prompt += f"\n{name}:\n"
                prompt += f"  Shape: {df.shape}\n"
                prompt += f"  Columns: {df.columns.tolist()}\n"
                prompt += f"  Data types:\n"
                for col, dtype in df.dtypes.items():
                    prompt += f"    {col}: {dtype}\n"
                prompt += f"\nFirst few rows:\n{df.head()}\n"

        if context:
            prompt += f"\nContext from previous analysis:\n{context}\n"

        prompt += """
Instructions:
1. Provide a clear analysis of what needs to be done
2. Write clean, executable Python code
3. Use pandas, numpy for data analysis
4. Include comments in your code
5. Wrap code in ```python blocks
6. Focus on statistical analysis, data cleaning, and insights
7. Store results in variables that can be accessed later
8. Don't create visualizations - that will be handled by another agent

Example:
```python
# Calculate summary statistics
summary = df.describe()
print(summary)
```

Provide your analysis and code:
"""
        return prompt

    def _extract_code_blocks(self, text: str) -> List[str]:
        code_blocks = []
        lines = text.split("\n")
        in_code_block = False
        current_block = []

        for line in lines:
            if line.strip().startswith("```python"):
                in_code_block = True
                current_block = []
            elif line.strip() == "```" and in_code_block:
                in_code_block = False
                if current_block:
                    code_blocks.append("\n".join(current_block))
            elif in_code_block:
                current_block.append(line)

        return code_blocks

    def _execute_code(self, code: str) -> Dict[str, Any]:
        exec_globals = {
            "pd": pd,
            "np": np,
            **{name: df for name, df in self.dataframes.items()},
        }

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        result = {
            "code": code,
            "success": False,
            "output": "",
            "error": None,
            "variables": {},
        }

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, exec_globals)

            result["success"] = True
            result["output"] = stdout_capture.getvalue()

            for key, value in exec_globals.items():
                if not key.startswith("_") and key not in ["pd", "np"]:
                    if isinstance(value, (int, float, str, bool, list, dict)):
                        result["variables"][key] = value
                    elif isinstance(value, pd.DataFrame):
                        result["variables"][key] = f"DataFrame{value.shape}"
                        self.dataframes[key] = value
                    elif isinstance(value, np.ndarray):
                        result["variables"][key] = f"Array{value.shape}"

        except Exception as e:
            result["error"] = str(e)
            result["output"] = stderr_capture.getvalue()
            result["traceback"] = traceback.format_exc()

        return result

    def _needs_visualization(self, query: str, results: List[Dict]) -> bool:
        
        viz_keywords = [
            "plot",
            "graph",
            "chart",
            "visualize",
            "show",
            "display",
            "trend",
            "distribution",
            "compare",
            "correlation",
        ]

        query_lower = query.lower()
        return any(keyword in query_lower for keyword in viz_keywords)
