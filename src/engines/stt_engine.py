"""
Speech-to-Text Engine for Whisper.cpp integration
"""
import subprocess
import logging
import os
from pathlib import Path
from typing import Optional

class WhisperCppEngine:
    def __init__(self, whisper_path: str, model_path: Optional[str] = None):
        """
        Initialize Whisper.cpp STT Engine
        
        Args:
            whisper_path: Path to whisper-cli.exe
            model_path: Path to whisper model (optional)
        """
        self.whisper_path = whisper_path
        self.model_path = model_path
        self.logger = logging.getLogger(__name__)
        
    def is_available(self) -> bool:
        """Check if Whisper.cpp is available"""
        try:
            if not os.path.exists(self.whisper_path):
                self.logger.error(f"Whisper executable not found: {self.whisper_path}")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error checking Whisper availability: {e}")
            return False
    
    def transcribe_file(self, audio_file: str) -> Optional[str]:
        """
        Transcribe audio file to text
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Transcribed text or None if failed
        """
        try:
            audio_path = Path(audio_file).resolve()
            if not audio_path.exists():
                self.logger.error(f"Audio file not found: {audio_path}")
                return None
            
            # Build command
            cmd = [str(Path(self.whisper_path).resolve())]
            
            # Add model if specified
            if self.model_path:
                cmd.extend(["-m", str(Path(self.model_path).resolve())])
            
            # Add audio file (use absolute path)
            cmd.extend(["-f", str(audio_path)])
            
            # Add output format
            cmd.extend(["--output-txt", "--no-timestamps"])
            
            self.logger.info(f"Running Whisper command: {' '.join(cmd)}")
            
            # Run whisper
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,  # 60 second timeout
                cwd=Path(self.whisper_path).parent
            )
            
            if result.returncode != 0:
                self.logger.error(f"Whisper failed: {result.stderr}")
                return None
            
            # Look for output text file (use proper path handling)
            txt_file = audio_path.parent / f"{audio_path.stem}.txt"
            
            if txt_file.exists():
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                # Clean up txt file
                txt_file.unlink(missing_ok=True)
                
                self.logger.info(f"Transcription successful: {text[:50]}...")
                return text
            else:
                self.logger.error("Whisper output file not found")
                return None
                
        except subprocess.TimeoutExpired:
            self.logger.error("Whisper transcription timed out")
            return None
        except Exception as e:
            self.logger.error(f"Error during transcription: {e}")
            return None
    
    def transcribe_bytes(self, audio_data: bytes, temp_dir: str = "temp") -> Optional[str]:
        """
        Transcribe audio data to text
        
        Args:
            audio_data: Raw audio data
            temp_dir: Temporary directory for files
            
        Returns:
            Transcribed text or None if failed
        """
        try:
            # Create temp directory
            temp_path = Path(temp_dir)
            temp_path.mkdir(exist_ok=True)
            
            # Save audio data to temporary file
            import time
            temp_file = temp_path / f"temp_audio_{int(time.time())}.wav"
            
            with open(temp_file, 'wb') as f:
                f.write(audio_data)
            
            # Transcribe
            result = self.transcribe_file(str(temp_file))
            
            # Clean up
            temp_file.unlink(missing_ok=True)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error transcribing audio data: {e}")
            return None