from typing import Dict, List, Any
from langchain_core.memory import BaseMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

class Memory(BaseMemory):
    
    history: ChatMessageHistory

    @property
    def memory_variables(self) -> List[str]:
        return ["history"]
    
    def __init__(self, k: int = 5):
        super().__init__()
        self.history = ChatMessageHistory()
        self.k = k
        
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "history": self.history.messages[-self.k*2 :]
        }
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        input_str = inputs.get("input", "")
        output_str = outputs.get("output", "")
        self.history.add_user_message(input_str)
        self.history.add_ai_message(output_str)
        
    def clear(self) -> None:
        self.history.clear()