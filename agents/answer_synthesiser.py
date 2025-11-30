import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import Dict, Any, List
import io
import sys
import traceback
import pandas as pd
import numpy as np
from contextlib import redirect_stdout, redirect_stderr
import ast
import os
import json
from .base_agent import BaseAgent, AgentResult, Message

class AnswerSynthesiserResult(BaseModel):
    final_answer: str = Field(description="The final answer to the user query")

class AnswerSynthesiserAgent(BaseAgent):
    """
    Agent responsible for:
    - Synthesizing answers 
    - Formatting answers into a readable format
    - Providing a summary of the answer 
    - Use this agent to synthesize the final answer. 
    - Use this agent to anser general questions and general chat questions.
    """

    def __init__(self, api_key: str):
        super().__init__(name="AnswerSynthesiser", api_key=api_key)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    def get_capabilities(self):
        return [
            "Answer general questions and conversational queries",
            "Synthesize final answers from analysis results",
            "Format responses with markdown for readability",
            "Provide summaries and insights in a clear, structured way",
            "Handle non-data-analysis queries (explanations, help, greetings)",
        ]

    def _build_prompt(self, query: str, context: Dict[str, Any]) -> str:
        
        has_code_results = context.get("codeinterpreter_data") is not None
        has_viz = context.get("visualizationagent_data") is not None
        has_presentation = context.get("presentationagent_data") is not None
        
        if has_code_results or has_viz or has_presentation:
            prompt = f"""You are an answer synthesizer agent. Your role is to create a clear, comprehensive response to the user's query based on the analysis performed by other agents.

User Query: {query}

"""
            if has_code_results:
                ci_data = context.get("codeinterpreter_data", {})
                prompt += "\n### Code Analysis Results:\n"
                if ci_data.get("analysis"):
                    prompt += f"{ci_data['analysis']}\n\n"
                if ci_data.get("results"):
                    prompt += "Execution Output:\n"
                    for result in ci_data["results"]:
                        if result.get("output"):
                            prompt += f"{result['output']}\n"
            
            if has_viz:
                viz_data = context.get("visualizationagent_data", {})
                viz_count = viz_data.get("visualization_count", 0)
                if viz_count > 0:
                    prompt += f"\n### Visualizations:\n{viz_count} visualizations were created.\n"
            
            if has_presentation:
                pres_data = context.get("presentationagent_data", {})
                if pres_data.get("presentation"):
                    prompt += f"\n### Report:\n{pres_data['presentation'].get('content', '')}\n"
            
            prompt += """
Instructions:
1. Synthesize a clear, concise answer to the user's query
2. Use markdown formatting for better readability
3. Include key findings, insights, and statistics
4. Structure your response with proper headings and bullet points
5. If there are numerical results, present them clearly
6. Be conversational but professional

Provide your synthesized answer in markdown format:
"""
        else:
            prompt = f"""You are a helpful AI assistant. Answer the user's question clearly and comprehensively.

User Query: {query}

Instructions:
1. Provide a clear, accurate answer to the question
2. Use markdown formatting for better readability
3. Include examples if helpful
4. Structure your response with headings and bullet points where appropriate
5. Be conversational but professional
6. If the question is about data analysis, explain that data should be uploaded first

Provide your answer in markdown format:
"""
        
        return prompt



    async def process(self, input_data: Dict[str, Any]) -> AgentResult:
        query = input_data.get("query", "")
        context = input_data.get("context", {})
        files = input_data.get("files", {})

        try:
            prompt = self._build_prompt(query, context)
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Structure the data for frontend display
            result_data = {
                "answer": response_text,
                "formatted_answer": response_text,  # Markdown formatted
                "query": query,
                "has_context": bool(context.get("codeinterpreter_data") or 
                                   context.get("visualizationagent_data") or 
                                   context.get("presentationagent_data"))
            }
            
            return AgentResult(
                success=True,
                data=result_data,
                message="Answer synthesized successfully",
                agent_name=self.name,
                next_agent=None,
            )
        except Exception as e:
            return AgentResult(
                success=False,
                data={"error": str(e)},
                message=f"Error synthesizing answer: {str(e)}",
                agent_name=self.name,
                next_agent=None,
            )