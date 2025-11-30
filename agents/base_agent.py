

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Message:

    role: str  # 'user', 'agent', 'system'
    content: str
    metadata: Dict[str, Any] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AgentResult:

    success: bool
    data: Any
    message: str
    agent_name: str
    next_agent: Optional[str] = None  # Which agent should handle this next
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseAgent(ABC):

    def __init__(self, name: str, api_key: str):
        self.name = name
        self.api_key = api_key
        self.conversation_history: List[Message] = []

    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> AgentResult:
        
        pass

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        pass

    def add_to_history(self, message: Message):
        
        self.conversation_history.append(message)

    def get_history(self) -> List[Message]:
        
        return self.conversation_history

    def clear_history(self):
        
        self.conversation_history = []
