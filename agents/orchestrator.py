"""
Agent Orchestrator - Manages multi-agent collaboration and workflow
"""

from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime
import google.generativeai as genai

from .base_agent import BaseAgent, AgentResult, Message
from .code_interpreter import CodeInterpreterAgent
from .visualization_agent import VisualizationAgent
from .presentation_agent import PresentationAgent
from .answer_synthesiser import AnswerSynthesiserAgent

class AgentOrchestrator:
    def __init__(self, api_key: str):
        self.api_key = api_key

        genai.configure(api_key=api_key)
        self.routing_model = genai.GenerativeModel("gemini-2.0-flash")

        self.agents: Dict[str, BaseAgent] = {
            "CodeInterpreter": CodeInterpreterAgent(api_key),
            "VisualizationAgent": VisualizationAgent(api_key),
            "PresentationAgent": PresentationAgent(api_key),
            "AnswerSynthesiser": AnswerSynthesiserAgent(api_key),
        }

        self.execution_history: List[Dict[str, Any]] = []
        self.current_context: Dict[str, Any] = {}

    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        return self.agents.get(agent_name)

    def list_agents(self) -> Dict[str, List[str]]:
        return {name: agent.get_capabilities() for name, agent in self.agents.items()}

    async def process_query(
        self,
        query: str,
        files: Optional[Dict[str, str]] = None,
        start_agent: str = "CodeInterpreter",
    ) -> Dict[str, Any]:
        
        if files is None:
            files = {}

        execution_start = datetime.now()

        input_data = {
            "query": query,
            "context": self.current_context.copy(),
            "files": files,
        }

        results = {
            "query": query,
            "execution_flow": [],
            "agent_results": {},
            "final_result": None,
            "success": True,
            "error": None,
        }

        current_agent_name = start_agent
        max_iterations = 10
        iteration = 0

        try:
            while current_agent_name and iteration < max_iterations:
                iteration += 1

                agent = self.get_agent(current_agent_name)
                if not agent:
                    raise ValueError(f"Agent {current_agent_name} not found")

                results["execution_flow"].append(
                    {
                        "agent": current_agent_name,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                agent_result = await agent.process(input_data)

                results["agent_results"][current_agent_name] = {
                    "success": agent_result.success,
                    "message": agent_result.message,
                    "data": agent_result.data,
                    "metadata": agent_result.metadata,
                }

                if not agent_result.success:
                    results["success"] = False
                    results["error"] = agent_result.message
                    break

                input_data["context"][f"{current_agent_name.lower()}_data"] = (
                    agent_result.data
                )

                current_agent_name = agent_result.next_agent

                input_data["query"] = query
                if iteration > 1:
                    input_data["files"] = {}

            if results["agent_results"]:
                last_agent = results["execution_flow"][-1]["agent"]
                results["final_result"] = results["agent_results"][last_agent]["data"]

            execution_end = datetime.now()
            self.execution_history.append(
                {
                    "timestamp": execution_start.isoformat(),
                    "duration": (execution_end - execution_start).total_seconds(),
                    "query": query,
                    "success": results["success"],
                    "agents_used": [
                        step["agent"] for step in results["execution_flow"]
                    ],
                }
            )

        except Exception as e:
            results["success"] = False
            results["error"] = str(e)

        return results

    async def chat(
        self,
        message: str,
        files: Optional[Dict[str, str]] = None,
        conversation_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        start_agent = self._determine_start_agent(message, files)
        if conversation_context:
            self.current_context.update(conversation_context)

        results = await self.process_query(message, files, start_agent)

        return results

    def _determine_start_agent(
        self, message: str, files: Optional[Dict[str, str]]
    ) -> str:
        if files:
            return "CodeInterpreter"

        context_info = []
        if self.current_context.get("dataframes"):
            context_info.append("- Data/DataFrames are already loaded in context")
        if self.current_context.get("codeinterpreter_data"):
            context_info.append("- Code analysis has been performed")
        if self.current_context.get("visualizationagent_data"):
            context_info.append("- Visualizations have been created")

        context_str = "\n".join(context_info) if context_info else "- No prior context"

        agent_descriptions = []
        for agent_name, agent in self.agents.items():
            capabilities = agent.get_capabilities()
            agent_descriptions.append(
                f"{agent_name}:\n  " + "\n  ".join(f"• {cap}" for cap in capabilities)
            )

        agents_str = "\n\n".join(agent_descriptions)

        prompt = f"""You are an intelligent agent router. Based on the user's query and current context, determine which agent should handle this request.

User Query: "{message}"

Current Context:
{context_str}

Available Agents:
{agents_str}

Instructions:
1. Analyze the user's query carefully
2. Consider the current context (what has already been done)
3. Choose the MOST APPROPRIATE agent to start processing this query
4. Respond with ONLY the agent name, nothing else

Rules:
- If the query requires data analysis/computation on new data, choose CodeInterpreter
- If data is already loaded and the query asks for visualization specifically, choose VisualizationAgent
- If analysis is done and the query asks for a detailed report/presentation, choose PresentationAgent
- If the query is a general question, explanation, or clarification, choose AnswerSynthesiser
- If the query asks about previous results/insights, choose AnswerSynthesiser
- If the query is conversational (hello, thanks, what can you do), choose AnswerSynthesiser
- DEFAULT: For simple questions or queries without data files, choose AnswerSynthesiser

Respond with ONLY ONE of these exact names:
- CodeInterpreter
- VisualizationAgent
- PresentationAgent
- AnswerSynthesiser

Your response (agent name only):"""

        try:
            response = self.routing_model.generate_content(prompt)
            selected_agent = response.text.strip()

            if selected_agent in self.agents:
                return selected_agent
            else:
                for agent_name in self.agents.keys():
                    if agent_name.lower() in selected_agent.lower():
                        return agent_name
                print(
                    f"Warning: Gemini returned invalid agent '{selected_agent}', defaulting to CodeInterpreter"
                )
                return "CodeInterpreter"

        except Exception as e:
            print(
                f"Warning: Error in agent selection ({str(e)}), defaulting to CodeInterpreter"
            )
            return "CodeInterpreter"

    def clear_context(self):
        self.current_context = {}
        for agent in self.agents.values():
            agent.clear_history()

    def get_execution_history(self) -> List[Dict[str, Any]]:
        return self.execution_history
