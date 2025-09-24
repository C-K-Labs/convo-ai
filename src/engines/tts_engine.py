"""
Text-to-Speech Engine for Piper TTS integration
"""
import subprocess
import logging
import os
import wave
from pathlib import Path
from typing import Optional
import threading

try:
    import sounddevice as sd
    import numpy as np
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    logging.warning("sounddevice not available, audio playback disabled")

class PiperTTSEngine:
    def __init__(self, 
                 piper_path: str,
                 voices_path: str,
                 default_voice: str = "Andrea"):
        """
        Initialize Piper TTS Engine
        
        Args:
            piper_path: Path to piper.exe
            voices_path: Path to voices directory
            default_voice: Default voice name
        """
        self.piper_path = piper_path
        self.voices_path = voices_path
        self.default_voice = default_voice
        self.logger = logging.getLogger(__name__)
        
    def is_available(self) -> bool:
        """Check if Piper TTS is available"""
        try:
            if not os.path.exists(self.piper_path):
                self.logger.error(f"Piper executable not found: {self.piper_path}")
                return False
                
            if not os.path.exists(self.voices_path):
                self.logger.error(f"Voices directory not found: {self.voices_path}")
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"Error checking Piper availability: {e}")
            return False
    
    def get_available_voices(self) -> list:
        """Get list of available voices"""
        try:
            voices_dir = Path(self.voices_path)
            if not voices_dir.exists():
                return []
                
            voices = []
            for voice_file in voices_dir.glob("*.onnx"):
                voices.append(voice_file.stem)
                
            return voices
        except Exception as e:
            self.logger.error(f"Error getting voices: {e}")
            return []
    
    def synthesize_to_file(self, 
                          text: str, 
                          output_file: str,
                          voice: Optional[str] = None) -> bool:
        """
        Synthesize text to audio file
        
        Args:
            text: Text to synthesize
            output_file: Output WAV file path
            voice: Voice name (optional, uses default)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not text or not text.strip():
                self.logger.warning("Empty text provided for synthesis")
                return False
            
            # Use default voice if not specified
            if not voice:
                voice = self.default_voice
            
            # Find voice model file
            voice_model = Path(self.voices_path) / f"{voice}.onnx"
            if not voice_model.exists():
                self.logger.error(f"Voice model not found: {voice_model}")
                # Try to find any available voice
                available_voices = self.get_available_voices()
                if available_voices:
                    voice = available_voices[0]
                    voice_model = Path(self.voices_path) / f"{voice}.onnx"
                    self.logger.info(f"Using fallback voice: {voice}")
                else:
                    self.logger.error("No voices available")
                    return False
            
            # Create output directory
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Build Piper command
            cmd = [
                self.piper_path,
                "--model", str(voice_model),
                "--output_file", str(output_file)
            ]
            
            self.logger.info(f"Running Piper command: {' '.join(cmd)}")
            
            # Run Piper TTS
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=Path(self.piper_path).parent
            )
            
            # Send text to stdin
            stdout, stderr = process.communicate(input=text, timeout=30)
            
            if process.returncode != 0:
                self.logger.error(f"Piper failed: {stderr}")
                return False
            
            # Check if output file was created
            if output_path.exists() and output_path.stat().st_size > 0:
                self.logger.info(f"TTS synthesis successful: {output_file}")
                return True
            else:
                self.logger.error("Piper output file not created or empty")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("Piper TTS timed out")
            process.kill()
            return False
        except Exception as e:
            self.logger.error(f"Error during TTS synthesis: {e}")
            return False
    
    def synthesize_to_bytes(self, 
                           text: str,
                           voice: Optional[str] = None,
                           temp_dir: str = "temp") -> Optional[bytes]:
        """
        Synthesize text to audio bytes
        
        Args:
            text: Text to synthesize
            voice: Voice name (optional)
            temp_dir: Temporary directory
            
        Returns:
            Audio data as bytes or None if failed
        """
        try:
            # Create temp directory
            temp_path = Path(temp_dir)
            temp_path.mkdir(exist_ok=True)
            
            # Create temporary file
            import time
            temp_file = temp_path / f"temp_tts_{int(time.time())}.wav"
            
            # Synthesize to file
            if self.synthesize_to_file(text, str(temp_file), voice):
                # Read file as bytes
                with open(temp_file, 'rb') as f:
                    audio_data = f.read()
                
                # Clean up
                temp_file.unlink(missing_ok=True)
                
                return audio_data
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error synthesizing to bytes: {e}")
            return None
    
    def play_audio_file(self, audio_file: str) -> bool:
        """
        Play audio file using sounddevice
        
        Args:
            audio_file: Path to WAV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not SOUNDDEVICE_AVAILABLE:
                self.logger.error("sounddevice not available for audio playback")
                return False
            
            if not os.path.exists(audio_file):
                self.logger.error(f"Audio file not found: {audio_file}")
                return False
            
            # Read WAV file
            with wave.open(audio_file, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
            
            # Convert to numpy array
            if sample_width == 1:
                dtype = np.uint8
            elif sample_width == 2:
                dtype = np.int16
            elif sample_width == 4:
                dtype = np.int32
            else:
                self.logger.error(f"Unsupported sample width: {sample_width}")
                return False
            
            audio_data = np.frombuffer(frames, dtype=dtype)
            
            # Reshape for channels
            if channels > 1:
                audio_data = audio_data.reshape(-1, channels)
            
            # Normalize for playback
            if dtype in [np.int16, np.int32]:
                audio_data = audio_data.astype(np.float32) / np.iinfo(dtype).max
            
            self.logger.info(f"Playing audio file: {audio_file}")
            
            # Play audio (blocking)
            sd.play(audio_data, sample_rate)
            sd.wait()  # Wait until finished
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error playing audio file: {e}")
            return False
    
    def play_audio_bytes(self, audio_data: bytes) -> bool:
        """
        Play audio from bytes
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save to temporary file and play
            import time
            temp_file = f"temp/play_temp_{int(time.time())}.wav"
            
            # Create temp directory
            Path(temp_file).parent.mkdir(exist_ok=True)
            
            with open(temp_file, 'wb') as f:
                f.write(audio_data)
            
            result = self.play_audio_file(temp_file)
            
            # Clean up
            Path(temp_file).unlink(missing_ok=True)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error playing audio bytes: {e}")
            return False