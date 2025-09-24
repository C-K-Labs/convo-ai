"""
Language Conversation AI - Main Launcher
Choose between text chat and voice conversation
"""
import sys
import yaml
import logging
from pathlib import Path

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'main.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def show_main_menu():
    """Show main application menu"""
    print("\n" + "=" * 60)
    print("🤖 Language Conversation AI")
    print("=" * 60)
    print("Choose your interaction mode:")
    print()
    print("1. 💬 Text Chat Mode")
    print("   - Keyboard-based text conversation")
    print("   - Fast testing and debugging")
    print()
    print("2. 🎤 Voice Conversation Mode")
    print("   - Real-time speech recognition and response")
    print("   - Natural conversation experience")
    print("   - VAD (Voice Activity Detection) support")
    print()
    print("3. ⚙️  System Information")
    print("4. 🚪 Exit")
    print("=" * 60)

def run_text_chat():
    """Launch text chat application"""
    print("\n🚀 Starting text chat mode...")
    
    try:
        sys.path.append(str(Path(__file__).parent))
        from chat_app import ChatApplication
        
        app = ChatApplication()
        if app.check_dependencies():
            print("✅ Text chat ready!")
            app.run_chat()
        else:
            print("❌ Cannot start text chat.")
            
    except Exception as e:
        print(f"❌ Error running text chat: {e}")
        logging.error(f"Text chat error: {e}")

def run_voice_conversation():
    """Launch voice conversation application"""
    print("\n🚀 Starting voice conversation mode...")
    
    try:
        sys.path.append(str(Path(__file__).parent))
        from voice_app import VoiceConversationApp
        
        app = VoiceConversationApp()
        app.run_interactive_menu()
        
    except Exception as e:
        print(f"❌ Error running voice conversation: {e}")
        logging.error(f"Voice conversation error: {e}")

def show_system_info():
    """Show system information and status"""
    print("\n" + "=" * 50)
    print("⚙️  System Information")
    print("=" * 50)
    
    try:
        config = load_config()
        
        print(f"📋 Configured model: {config['models']['llm']}")
        print(f"🌐 Ollama host: {config['ollama']['host']}:{config['ollama']['port']}")
        print(f"🎤 STT engine: {config['stt']['engine']}")
        print(f"🔊 TTS engine: {config['tts']['engine']}")
        print(f"📝 Log level: {config['app']['log_level']}")
        
        print("\n🔍 Dependency check:")
        
        # Check Ollama
        sys.path.append(str(Path(__file__).parent))
        from engines.llm_engine import OllamaLLMEngine
        
        llm = OllamaLLMEngine(
            host=config['ollama']['host'],
            port=config['ollama']['port']
        )
        
        if llm.is_available():
            models = llm.list_models()
            print(f"✅ Ollama: Running ({len(models)} models available)")
            for model in models[:3]:  # Show first 3 models
                print(f"   - {model['name']}")
            if len(models) > 3:
                print(f"   ... and {len(models) - 3} more")
        else:
            print("❌ Ollama: Connection failed")
        
        # Check STT
        from engines.stt_engine import WhisperCppEngine
        stt = WhisperCppEngine(
            whisper_path=config['stt']['whisper_cpp_path'],
            model_path=config['stt'].get('model_path')
        )
        
        if stt.is_available():
            print("✅ Whisper STT: Available")
        else:
            print("❌ Whisper STT: Not available")
        
        # Check TTS
        from engines.tts_engine import PiperTTSEngine
        tts = PiperTTSEngine(
            piper_path=config['tts']['piper_path'],
            voices_path=config['tts']['voices_path'],
            default_voice=config['tts']['default_voice']
        )
        
        if tts.is_available():
            print("✅ Piper TTS: Available")
        else:
            print("❌ Piper TTS: Not available")
        
        # Check audio devices
        from engines.audio_recorder import RealTimeAudioRecorder
        recorder = RealTimeAudioRecorder()
        devices = recorder.get_audio_devices()
        recorder.cleanup()
        
        print(f"🎤 Audio input devices: {len(devices)} found")
        
        print("\n📁 Directory structure:")
        important_dirs = ['logs', 'temp', 'conversations']
        for dir_name in important_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists():
                print(f"✅ {dir_name}/")
            else:
                print(f"📁 {dir_name}/ (will be created)")
                dir_path.mkdir(exist_ok=True)
        
    except Exception as e:
        print(f"❌ Error checking system info: {e}")
        logging.error(f"System info error: {e}")

def main():
    """Main application entry point"""
    config = load_config()
    setup_logging(config['app']['log_level'])
    
    logger = logging.getLogger(__name__)
    logger.info("Language Conversation AI started")
    
    # Create necessary directories
    for dir_name in ['logs', 'temp', 'conversations']:
        Path(dir_name).mkdir(exist_ok=True)
    
    while True:
        try:
            show_main_menu()
            choice = input("\nChoose option (1-4): ").strip()
            
            if choice == '1':
                run_text_chat()
            elif choice == '2':
                run_voice_conversation()
            elif choice == '3':
                show_system_info()
            elif choice == '4':
                print("\n👋 Exiting Language Conversation AI.")
                logger.info("Application terminated by user")
                break
            else:
                print("❌ Invalid choice. Please select 1-4.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Exiting Language Conversation AI.")
            logger.info("Application terminated by KeyboardInterrupt")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()