"""
LLM Engine for Ollama integration
"""
import requests
import json
import logging
from typing import Dict, Any, Optional

class OllamaLLMEngine:
    def __init__(self, host: str = "localhost", port: int = 11434):
        self.base_url = f"http://{host}:{port}"
        self.logger = logging.getLogger(__name__)
        
    def is_available(self) -> bool:
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def list_models(self) -> list:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                return response.json().get('models', [])
        except requests.RequestException as e:
            self.logger.error(f"Failed to get models: {e}")
        return []
    
    def generate_response(self, model: str, prompt: str, **kwargs) -> Optional[str]:
        """Generate response from LLM"""
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get('response', '').strip()
            else:
                self.logger.error(f"API error: {response.status_code}")
                
        except requests.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            
        return None
    
    def chat(self, model: str, messages: list, **kwargs) -> Optional[str]:
        """Chat with conversation context"""
        data = {
            "model": model,
            "messages": messages,
            "stream": False,
            **kwargs
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get('message', {}).get('content', '').strip()
            else:
                self.logger.error(f"Chat API error: {response.status_code}")
                
        except requests.RequestException as e:
            self.logger.error(f"Chat request failed: {e}")
            
        return None
