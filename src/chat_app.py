"""
Main Chat Application
"""
import sys
import yaml
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from engines.llm_engine import OllamaLLMEngine
from utils.conversation_manager import ConversationManager

class ChatApplication:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self.load_config(config_path)
        self.setup_logging()
        
        # Initialize components
        self.llm_engine = OllamaLLMEngine(
            host=self.config['ollama']['host'],
            port=self.config['ollama']['port']
        )
        self.conversation_manager = ConversationManager()
        self.logger = logging.getLogger(__name__)
        
        # Set system prompt for language learning
        self.system_prompt = """You are a helpful language conversation partner. 
You can speak both Korean and English fluently. 
Help users practice languages through natural conversation.
Be encouraging and provide gentle corrections when needed."""
        
    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config['app']['log_level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'app.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def check_dependencies(self) -> bool:
        """Check if all required services are available"""
        if not self.llm_engine.is_available():
            print("❌ Ollama service is not running!")
            print("Please start Ollama and try again.")
            return False
            
        models = self.llm_engine.list_models()
        model_names = [m['name'] for m in models]
        
        if self.config['models']['llm'] not in model_names:
            print(f"❌ Model {self.config['models']['llm']} is not available!")
            print(f"Available models: {model_names}")
            return False
            
        return True
    
    def get_ai_response(self, user_message: str) -> str:
        """Get response from AI model"""
        # Add user message to conversation
        self.conversation_manager.add_message("user", user_message)
        
        # Prepare messages for chat API
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_manager.get_chat_messages())
        
        # Get AI response
        response = self.llm_engine.chat(
            model=self.config['models']['llm'],
            messages=messages
        )
        
        if response:
            self.conversation_manager.add_message("assistant", response)
            return response
        else:
            return "Sorry, I couldn't generate a response. Please try again."
    
    def run_chat(self):
        """Run interactive chat session"""
        print("🤖 Language Conversation AI")
        print("=" * 50)
        print("Type 'quit' to exit, 'clear' to clear history")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    self.conversation_manager.clear_history()
                    print("🧹 Conversation history cleared!")
                    continue
                elif not user_input:
                    continue
                
                print("🤖 AI: ", end="", flush=True)
                response = self.get_ai_response(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                self.logger.error(f"Error in chat loop: {e}")
                print(f"❌ Error: {e}")

def main():
    app = ChatApplication()
    
    if not app.check_dependencies():
        return
        
    print("✅ All dependencies are ready!")
    app.run_chat()

if __name__ == "__main__":
    main()
