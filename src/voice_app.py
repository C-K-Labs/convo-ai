"""
Voice Conversation Application
Real-time speech-to-speech AI conversation
"""
import sys
import yaml
import logging
import threading
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from engines.llm_engine import OllamaLLMEngine
from engines.stt_engine import WhisperCppEngine
from engines.tts_engine import PiperTTSEngine
from engines.audio_recorder import RealTimeAudioRecorder
from utils.conversation_manager import ConversationManager

class VoiceConversationApp:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize voice conversation application"""
        self.config = self.load_config(config_path)
        self.setup_logging()
        
        # Initialize engines
        self.llm_engine = OllamaLLMEngine(
            host=self.config['ollama']['host'],
            port=self.config['ollama']['port']
        )
        
        self.stt_engine = WhisperCppEngine(
            whisper_path=self.config['stt']['whisper_cpp_path'],
            model_path=self.config['stt'].get('model_path')
        )
        
        self.tts_engine = PiperTTSEngine(
            piper_path=self.config['tts']['piper_path'],
            voices_path=self.config['tts']['voices_path'],
            default_voice=self.config['tts']['default_voice']
        )
        
        self.audio_recorder = RealTimeAudioRecorder(
            sample_rate=self.config['audio']['sample_rate'],
            channels=self.config['audio']['channels'],
            chunk_size=self.config['audio']['chunk_size'],
            energy_threshold=0.005,  # Adjust based on environment
            silence_duration=2.0,     # 2 seconds of silence to end speech
            max_speech_duration=30.0  # Max 30 seconds per speech
        )
        
        self.conversation_manager = ConversationManager()
        self.logger = logging.getLogger(__name__)
        
        # System prompt for language learning
        self.system_prompt = """You are a helpful language conversation partner. 
You can speak both Korean and English fluently. 
Help users practice languages through natural conversation.
Be encouraging and provide gentle corrections when needed.
Keep responses conversational and not too long."""
        
        # Conversation state
        self.is_conversation_active = False
        self.is_processing = False
        self.current_energy = 0.0
        
        # Setup callbacks
        self.setup_audio_callbacks()
        
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
                logging.FileHandler(log_dir / 'voice_app.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def setup_audio_callbacks(self):
        """Setup audio recorder callbacks"""
        self.audio_recorder.set_callbacks(
            on_speech_start=self.on_speech_start,
            on_speech_data=self.on_speech_data,
            on_speech_end=self.on_speech_end,
            on_energy_update=self.on_energy_update
        )
    
    def on_speech_start(self):
        """Called when user starts speaking"""
        if not self.is_conversation_active:
            return
            
        print("\n🎤 Listening...")
        self.logger.info("Speech started")
    
    def on_speech_data(self, audio_data: bytes):
        """Called during speech (real-time data)"""
        # For future: real-time STT streaming could be implemented here
        pass
    
    def on_speech_end(self, audio_data: bytes):
        """Called when user stops speaking"""
        if not self.is_conversation_active or self.is_processing:
            return
            
        print("🔄 Processing...")
        self.is_processing = True
        
        # Process in separate thread to avoid blocking
        processing_thread = threading.Thread(
            target=self.process_speech,
            args=(audio_data,),
            daemon=True
        )
        processing_thread.start()
    
    def on_energy_update(self, energy: float):
        """Called with real-time energy levels"""
        self.current_energy = energy
        # Visual indicator could be added here
    
    def process_speech(self, audio_data: bytes):
        """Process captured speech through STT -> LLM -> TTS pipeline"""
        try:
            start_time = time.time()
            
            # Save audio temporarily for STT processing
            temp_audio_file = f"temp/speech_{int(time.time())}.wav"
            self.audio_recorder.save_audio_to_file(audio_data, temp_audio_file)
            
            # Step 1: Speech to Text
            print("📝 Converting speech to text...")
            user_text = self.stt_engine.transcribe_file(temp_audio_file)
            
            if not user_text or not user_text.strip():
                print("❌ Could not recognize speech. Please try again.")
                self.is_processing = False
                return
                
            print(f"👤 User: {user_text}")
            
            # Add to conversation history
            self.conversation_manager.add_message("user", user_text)
            
            # Step 2: Get AI response
            print("🤖 Generating AI response...")
            ai_response = self.get_ai_response(user_text)
            
            if not ai_response:
                print("❌ Could not generate AI response.")
                self.is_processing = False
                return
                
            print(f"🤖 AI: {ai_response}")
            
            # Step 3: Text to Speech
            print("🔊 Converting to speech...")
            audio_output_file = f"temp/response_{int(time.time())}.wav"
            
            if self.tts_engine.synthesize_to_file(ai_response, audio_output_file):
                # Play the audio (this will block until playback is complete)
                self.tts_engine.play_audio_file(audio_output_file)
            else:
                print("❌ Speech synthesis failed.")
            
            # Clean up temporary files
            Path(temp_audio_file).unlink(missing_ok=True)
            Path(audio_output_file).unlink(missing_ok=True)
            
            processing_time = time.time() - start_time
            self.logger.info(f"Speech processing completed in {processing_time:.2f} seconds")
            
            print("\n🎤 Please speak again...")
            
        except Exception as e:
            self.logger.error(f"Error processing speech: {e}")
            print(f"❌ Error during processing: {e}")
            
        finally:
            self.is_processing = False
    
    def get_ai_response(self, user_message: str) -> str:
        """Get response from AI model"""
        try:
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
                return "Sorry, I couldn't generate a response."
                
        except Exception as e:
            self.logger.error(f"Error getting AI response: {e}")
            return "Sorry, an error occurred while generating the response."
    
    def check_dependencies(self) -> bool:
        """Check if all required services are available"""
        print("🔍 Checking system...")
        
        # Check Ollama
        if not self.llm_engine.is_available():
            print("❌ Ollama service is not running!")
            print("Command: ollama serve")
            return False
        
        models = self.llm_engine.list_models()
        model_names = [m['name'] for m in models]
        
        if self.config['models']['llm'] not in model_names:
            print(f"❌ Model {self.config['models']['llm']} not found!")
            print(f"Available models: {model_names}")
            return False
            
        # Check STT
        if not self.stt_engine.is_available():
            print("❌ Whisper STT engine is not available!")
            return False
            
        # Check TTS
        if not self.tts_engine.is_available():
            print("❌ Piper TTS engine is not available!")
            return False
        
        # Check audio devices
        devices = self.audio_recorder.get_audio_devices()
        if not devices:
            print("❌ No audio input devices available!")
            return False
            
        print("✅ All systems are operational!")
        return True
    
    def start_conversation(self):
        """Start voice conversation"""
        if self.is_conversation_active:
            print("Conversation is already active.")
            return
            
        self.is_conversation_active = True
        
        print("\n🎤 Starting voice conversation!")
        print("=" * 50)
        print("💡 Tip: Speak naturally. AI will respond after 2 seconds of silence.")
        print("💡 Press 'Ctrl+C' to exit.")
        print("=" * 50)
        
        # Start listening
        self.audio_recorder.start_listening()
        
        print("🎤 Listening...")
        
        try:
            # Keep the conversation running
            while self.is_conversation_active:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n👋 Ending conversation.")
            
        finally:
            self.stop_conversation()
    
    def stop_conversation(self):
        """Stop voice conversation"""
        if not self.is_conversation_active:
            return
            
        self.is_conversation_active = False
        self.audio_recorder.stop_listening()
        
        print("🔇 Voice conversation ended.")
        
        # Save conversation if desired
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversation_manager.save_conversation(f"conversations/voice_chat_{timestamp}.json")
    
    def run_interactive_menu(self):
        """Run interactive menu"""
        print("\n🤖 Voice Conversation AI")
        print("=" * 50)
        
        while True:
            print("\nChoose an option:")
            print("1. Start voice conversation")
            print("2. Check audio devices")
            print("3. Check system status")
            print("4. Exit")
            
            try:
                choice = input("\nChoice (1-4): ").strip()
                
                if choice == '1':
                    if self.check_dependencies():
                        self.start_conversation()
                elif choice == '2':
                    self.show_audio_devices()
                elif choice == '3':
                    self.check_dependencies()
                elif choice == '4':
                    print("👋 Exiting application.")
                    break
                else:
                    print("❌ Invalid choice.")
                    
            except KeyboardInterrupt:
                print("\n👋 Exiting application.")
                break
    
    def show_audio_devices(self):
        """Show available audio devices"""
        print("\n🎤 Available audio input devices:")
        devices = self.audio_recorder.get_audio_devices()
        
        for device in devices:
            print(f"  {device['index']}: {device['name']}")
            print(f"      Channels: {device['channels']}, Sample Rate: {device['sample_rate']}")
    
    def cleanup(self):
        """Clean up resources"""
        if self.is_conversation_active:
            self.stop_conversation()
            
        self.audio_recorder.cleanup()
        
        # Create temp directory for cleanup
        temp_dir = Path("temp")
        if temp_dir.exists():
            for file in temp_dir.glob("*.wav"):
                file.unlink(missing_ok=True)

def main():
    """Main application entry point"""
    app = VoiceConversationApp()
    
    try:
        app.run_interactive_menu()
    finally:
        app.cleanup()

if __name__ == "__main__":
    main()