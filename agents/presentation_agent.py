

import google.generativeai as genai
from typing import Dict, Any, List
import json
from datetime import datetime

from .base_agent import BaseAgent, AgentResult, Message


class PresentationAgent(BaseAgent):
    

    def __init__(self, api_key: str):
        super().__init__(name="PresentationAgent", api_key=api_key)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    def get_capabilities(self) -> List[str]:
        return [
            "Generate structured reports",
            "Create presentation summaries",
            "Format insights and findings",
            "Create executive summaries",
            "Generate markdown reports",
        ]

    async def process(self, input_data: Dict[str, Any]) -> AgentResult:
        
        query = input_data.get("query", "")
        context = input_data.get("context", {})

        prompt = self._build_prompt(query, context)

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text

            presentation = self._structure_presentation(response_text, context)

            return AgentResult(
                success=True,
                data={
                    "presentation": presentation,
                    "raw_text": response_text,
                    "timestamp": datetime.now().isoformat(),
                },
                message="Presentation generated successfully",
                agent_name=self.name,
                next_agent=None,
            )

        except Exception as e:
            return AgentResult(
                success=False,
                data={"error": str(e)},
                message=f"Error creating presentation: {str(e)}",
                agent_name=self.name,
            )

    def _build_prompt(self, query: str, context: Dict[str, Any]) -> str:
        prompt = f"""You are a presentation and report writing expert. Create a comprehensive, well-structured report.

User Query: {query}

"""

        if "code_interpreter_data" in context:
            prompt += "Analysis Results:\n"
            ci_data = context["code_interpreter_data"]
            if "analysis" in ci_data:
                prompt += f"{ci_data['analysis']}\n\n"
            if "results" in ci_data and ci_data["results"]:
                prompt += "Execution Results:\n"
                for result in ci_data["results"]:
                    if result.get("output"):
                        prompt += f"{result['output']}\n"

        if "visualization_data" in context:
            viz_data = context["visualization_data"]
            viz_count = viz_data.get("visualization_count", 0)
            if viz_count > 0:
                prompt += f"\nVisualizations Created: {viz_count}\n"
                if "analysis" in viz_data:
                    prompt += f"{viz_data['analysis']}\n"

        prompt += """
Instructions:
1. Create a professional, structured report/presentation
2. Use clear markdown formatting with headers
3. Include the following sections:
   - Executive Summary
   - Key Findings
   - Detailed Analysis
   - Insights and Recommendations
   - Conclusion
4. Use bullet points and numbered lists where appropriate
5. Highlight important metrics and statistics
6. Make it easy to understand for both technical and non-technical audiences
7. Reference visualizations where applicable

Format your response in markdown with clear sections.

Generate the presentation:
"""
        return prompt

    def _structure_presentation(
        self, text: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        presentation = {
            "title": "Data Analysis Report",
            "generated_at": datetime.now().isoformat(),
            "content": text,
            "sections": [],
            "visualizations": [],
            "metadata": {},
        }

        if "visualization_data" in context:
            viz_data = context["visualization_data"]
            if "visualizations" in viz_data:
                presentation["visualizations"] = viz_data["visualizations"]

        lines = text.split("\n")
        current_section = None

        for line in lines:
            if line.startswith("# "):
                presentation["title"] = line.replace("# ", "").strip()
            elif line.startswith("## "):
                if current_section:
                    presentation["sections"].append(current_section)
                current_section = {
                    "title": line.replace("## ", "").strip(),
                    "content": [],
                }
            elif current_section:
                current_section["content"].append(line)

        if current_section:
            presentation["sections"].append(current_section)

        presentation["metadata"] = {
            "num_sections": len(presentation["sections"]),
            "num_visualizations": len(presentation["visualizations"]),
            "has_code_analysis": "code_interpreter_data" in context,
        }

        return presentation
