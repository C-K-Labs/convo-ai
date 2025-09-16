"""
Conversation Manager for handling chat history and context
"""
import json
import logging
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

class ConversationManager:
    def __init__(self, max_history: int = 10):
        self.conversation_history = []
        self.max_history = max_history
        self.logger = logging.getLogger(__name__)
        
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add message to conversation history"""
        message = {
            "role": role,  # "user" or "assistant"
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.conversation_history.append(message)
        
        # Keep only recent messages within limit
        if len(self.conversation_history) > self.max_history * 2:  # *2 for user+assistant pairs
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
            
        self.logger.debug(f"Added {role} message: {content[:50]}...")
    
    def get_chat_messages(self) -> List[Dict[str, str]]:
        """Get messages in Ollama chat format"""
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.conversation_history
        ]
    
    def get_context_prompt(self) -> str:
        """Get conversation history as a single prompt"""
        if not self.conversation_history:
            return ""
            
        context = "Previous conversation:\n"
        for msg in self.conversation_history[-6:]:  # Last 6 messages
            role_display = "Human" if msg["role"] == "user" else "Assistant"
            context += f"{role_display}: {msg['content']}\n"
            
        return context + "\nCurrent conversation:\n"
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        self.logger.info("Conversation history cleared")
    
    def save_conversation(self, filepath: str):
        """Save conversation to file"""
        data = {
            "conversation": self.conversation_history,
            "saved_at": datetime.now().isoformat()
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"Conversation saved to {filepath}")
    
    def load_conversation(self, filepath: str):
        """Load conversation from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.conversation_history = data.get('conversation', [])
                self.logger.info(f"Conversation loaded from {filepath}")
        except FileNotFoundError:
            self.logger.warning(f"Conversation file not found: {filepath}")
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON in conversation file: {filepath}")
