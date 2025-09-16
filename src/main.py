"""
Language Conversation AI - Main Application
"""
import yaml
import logging
from pathlib import Path

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main application entry point"""
    config = load_config()
    setup_logging(config['app']['log_level'])
    
    logger = logging.getLogger(__name__)
    logger.info("Language Conversation AI starting...")
    logger.info(f"Using LLM model: {config['models']['llm']}")
    
    print("🤖 Language Conversation AI")
    print("=" * 50)
    print("Basic setup complete! Ready for development.")

if __name__ == "__main__":
    main()
