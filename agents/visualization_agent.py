

import google.generativeai as genai
from typing import Dict, Any, List
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io
import base64
from pathlib import Path
import json

from .base_agent import BaseAgent, AgentResult, Message


class VisualizationAgent(BaseAgent):

    def __init__(self, api_key: str):
        super().__init__(name="VisualizationAgent", api_key=api_key)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        self.visualizations = []

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (10, 6)

    def get_capabilities(self) -> List[str]:
        return [
            "Create matplotlib/seaborn visualizations",
            "Generate charts and graphs",
            "Create distribution plots",
            "Generate correlation heatmaps",
            "Create time series plots",
        ]

    async def process(self, input_data: Dict[str, Any]) -> AgentResult:
        query = input_data.get("query", "")
        context = input_data.get("context", {})

        prompt = self._build_prompt(query, context)

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text

            code_blocks = self._extract_code_blocks(response_text)

            if not code_blocks:
                return AgentResult(
                    success=True,
                    data={"analysis": response_text, "visualizations": []},
                    message="No visualizations needed",
                    agent_name=self.name,
                    next_agent="AnswerSynthesiser",
                )

            visualizations = []
            for code in code_blocks:
                viz_result = self._create_visualization(code, context)
                if viz_result["success"]:
                    visualizations.append(viz_result)

            return AgentResult(
                success=True,
                data={
                    "analysis": response_text,
                    "visualizations": visualizations,
                    "visualization_count": len(visualizations),
                },
                message=f"Created {len(visualizations)} visualizations",
                agent_name=self.name,
                next_agent="AnswerSynthesiser",
                metadata={"has_visualizations": len(visualizations) > 0},
            )

        except Exception as e:
            return AgentResult(
                success=False,
                data={"error": str(e)},
                message=f"Error creating visualizations: {str(e)}",
                agent_name=self.name,
            )

    def _build_prompt(self, query: str, context: Dict[str, Any]) -> str:
        prompt = f"""You are a data visualization expert using matplotlib and seaborn.

User Query: {query}

Context from Code Interpreter:
{json.dumps(context, indent=2, default=str)}

Instructions:
1. Analyze what visualizations would best answer the user's query
2. Write Python code using matplotlib and seaborn
3. Use the data provided in the context
4. Create clear, informative visualizations
5. Add proper titles, labels, and legends
6. Wrap code in ```python blocks
7. Save each plot using plt.savefig('plot.png', bbox_inches='tight', dpi=150)
8. Close plots with plt.close() after saving

Available data:
"""

        if "dataframes_info" in context:
            prompt += "\nDataFrames:\n"
            for name, info in context["dataframes_info"].items():
                prompt += f"  {name}: {info}\n"

        prompt += """
Example:
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Create visualization
plt.figure(figsize=(10, 6))
# ... your plotting code ...
plt.title('My Chart Title')
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.savefig('plot.png', bbox_inches='tight', dpi=150)
plt.close()
```

Provide your visualization code:
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

    def _create_visualization(
        self, code: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:

        exec_globals = {
            "plt": plt,
            "sns": sns,
            "pd": pd,
            "np": np,
        }

        if "dataframes" in context:
            for name, data in context["dataframes"].items():
                if isinstance(data, dict):
                    exec_globals[name] = pd.DataFrame(data)
                else:
                    exec_globals[name] = data

        result = {
            "code": code,
            "success": False,
            "image_base64": None,
            "error": None,
            "description": "",
        }

        try:
            exec(code, exec_globals)

            if Path("plot.png").exists():
                with open("plot.png", "rb") as img_file:
                    img_data = img_file.read()
                    result["image_base64"] = base64.b64encode(img_data).decode("utf-8")

                result["success"] = True
                result["description"] = "Visualization created successfully"

                Path("plot.png").unlink()
            else:
                result["error"] = "No plot file was created"

        except Exception as e:
            result["error"] = str(e)

        plt.close("all")

        return result
