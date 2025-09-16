"""
Test LLM Engine functionality
"""
import sys
sys.path.append('src')

from engines.llm_engine import OllamaLLMEngine
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize LLM engine
llm = OllamaLLMEngine(
    host=config['ollama']['host'],
    port=config['ollama']['port']
)

print("🔍 Testing Ollama connection...")
if llm.is_available():
    print("✅ Ollama is running!")
    
    # Test model list
    models = llm.list_models()
    print(f"📋 Available models: {[m['name'] for m in models]}")
    
    # Test simple generation
    print("\n🤖 Testing LLM response...")
    response = llm.generate_response(
        model=config['models']['llm'],
        prompt="Hello! Please respond in Korean and English."
    )
    
    if response:
        print(f"🎉 LLM Response: {response}")
    else:
        print("❌ Failed to get response")
        
else:
    print("❌ Ollama is not running. Please start Ollama service.")
