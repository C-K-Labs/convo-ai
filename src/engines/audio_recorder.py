"""
Real-time Audio Recording with Voice Activity Detection (VAD)
Using sounddevice instead of pyaudio
"""
import wave
import threading
import time
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Callable
from collections import deque
import sounddevice as sd

class VoiceActivityDetector:
    def __init__(self, 
                 sample_rate: int = 16000,
                 frame_duration: float = 0.03,  # 30ms frames
                 energy_threshold: float = 0.01,
                 silence_duration: float = 2.0):
        """
        Voice Activity Detection
        
        Args:
            sample_rate: Audio sample rate
            frame_duration: Frame duration in seconds
            energy_threshold: Energy threshold for voice detection
            silence_duration: Silence duration to consider speech ended
        """
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration)
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        
        self.silence_frames_needed = int(silence_duration / frame_duration)
        self.silence_count = 0
        self.is_speaking = False
        self.speech_started = False
        
        self.logger = logging.getLogger(__name__)
    
    def calculate_energy(self, audio_data: np.ndarray) -> float:
        """Calculate audio energy level"""
        try:
            if len(audio_data) == 0:
                return 0.0
            
            # Calculate RMS energy
            energy = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
            return energy
            
        except Exception as e:
            self.logger.error(f"Error calculating energy: {e}")
            return 0.0
    
    def process_frame(self, audio_data: np.ndarray) -> dict:
        """
        Process audio frame for voice activity
        
        Returns:
            dict: {
                'energy': float,
                'is_voice': bool,
                'speech_started': bool,
                'speech_ended': bool
            }
        """
        energy = self.calculate_energy(audio_data)
        is_voice = energy > self.energy_threshold
        
        speech_started = False
        speech_ended = False
        
        if is_voice:
            # Voice detected
            self.silence_count = 0
            if not self.is_speaking:
                self.is_speaking = True
                if not self.speech_started:
                    self.speech_started = True
                    speech_started = True
                    self.logger.debug("Speech started")
        else:
            # Silence detected
            if self.is_speaking:
                self.silence_count += 1
                if self.silence_count >= self.silence_frames_needed:
                    # End of speech
                    self.is_speaking = False
                    speech_ended = True
                    self.logger.debug("Speech ended")
        
        return {
            'energy': energy,
            'is_voice': is_voice,
            'speech_started': speech_started,
            'speech_ended': speech_ended,
            'is_speaking': self.is_speaking
        }
    
    def reset(self):
        """Reset VAD state"""
        self.silence_count = 0
        self.is_speaking = False
        self.speech_started = False

class RealTimeAudioRecorder:
    def __init__(self,
                 sample_rate: int = 16000,
                 channels: int = 1,
                 chunk_size: int = 1024,
                 dtype = np.int16,
                 energy_threshold: float = 0.01,
                 silence_duration: float = 2.0,
                 max_speech_duration: float = 30.0):
        """
        Real-time audio recorder with VAD using sounddevice
        
        Args:
            sample_rate: Audio sample rate (Hz)
            channels: Number of audio channels
            chunk_size: Audio buffer size
            dtype: Audio data type
            energy_threshold: VAD energy threshold
            silence_duration: Silence duration to end speech
            max_speech_duration: Maximum speech duration
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.max_speech_duration = max_speech_duration
        
        self.stream = None
        self.is_recording = False
        self.recording_thread = None
        self.audio_queue = deque()
        
        # VAD setup
        self.vad = VoiceActivityDetector(
            sample_rate=sample_rate,
            energy_threshold=energy_threshold,
            silence_duration=silence_duration
        )
        
        # Audio buffers
        self.speech_buffer = deque()
        self.speech_frames = []
        self.is_collecting_speech = False
        
        # Callbacks
        self.on_speech_start = None
        self.on_speech_data = None
        self.on_speech_end = None
        self.on_energy_update = None
        
        self.logger = logging.getLogger(__name__)
        
    def set_callbacks(self,
                     on_speech_start: Optional[Callable] = None,
                     on_speech_data: Optional[Callable] = None,
                     on_speech_end: Optional[Callable[[bytes], None]] = None,
                     on_energy_update: Optional[Callable[[float], None]] = None):
        """Set callback functions"""
        self.on_speech_start = on_speech_start
        self.on_speech_data = on_speech_data
        self.on_speech_end = on_speech_end
        self.on_energy_update = on_energy_update
        
    def audio_callback(self, indata, frames, time, status):
        """Callback function for sounddevice"""
        if status:
            self.logger.warning(f"Audio callback status: {status}")
        
        # Add audio data to queue
        self.audio_queue.append(indata.copy())
        
    def start_listening(self):
        """Start continuous audio monitoring"""
        if self.is_recording:
            self.logger.warning("Already listening")
            return
            
        try:
            # Start audio stream
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=self.chunk_size,
                callback=self.audio_callback
            )
            
            self.stream.start()
            self.is_recording = True
            self.vad.reset()
            
            self.recording_thread = threading.Thread(
                target=self._listening_loop,
                daemon=True
            )
            self.recording_thread.start()
            
            self.logger.info("Started listening for speech")
            
        except Exception as e:
            self.logger.error(f"Failed to start listening: {e}")
            raise
    
    def _listening_loop(self):
        """Main listening loop with VAD"""
        speech_start_time = None
        
        while self.is_recording:
            try:
                # Wait for audio data
                if not self.audio_queue:
                    time.sleep(0.01)
                    continue
                
                # Get audio data
                audio_data = self.audio_queue.popleft()
                
                # Flatten if stereo
                if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                    audio_data = np.mean(audio_data, axis=1)
                else:
                    audio_data = audio_data.flatten()
                
                # Process with VAD
                vad_result = self.vad.process_frame(audio_data)
                
                # Update energy callback
                if self.on_energy_update:
                    self.on_energy_update(vad_result['energy'])
                
                # Handle speech start
                if vad_result['speech_started']:
                    self.is_collecting_speech = True
                    self.speech_frames = []
                    speech_start_time = time.time()
                    
                    if self.on_speech_start:
                        self.on_speech_start()
                
                # Collect speech data
                if self.is_collecting_speech:
                    self.speech_frames.append(audio_data)
                    
                    # Send real-time speech data (convert to bytes)
                    if self.on_speech_data:
                        audio_bytes = audio_data.astype(self.dtype).tobytes()
                        self.on_speech_data(audio_bytes)
                    
                    # Check for maximum duration
                    if speech_start_time and (time.time() - speech_start_time > self.max_speech_duration):
                        self.logger.warning("Maximum speech duration reached")
                        vad_result['speech_ended'] = True
                
                # Handle speech end
                if vad_result['speech_ended'] and self.is_collecting_speech:
                    self.is_collecting_speech = False
                    
                    # Combine all speech frames
                    if self.speech_frames:
                        complete_speech_array = np.concatenate(self.speech_frames)
                        complete_speech_bytes = complete_speech_array.astype(self.dtype).tobytes()
                        
                        if self.on_speech_end:
                            self.on_speech_end(complete_speech_bytes)
                        
                        self.logger.info(f"Speech captured: {len(complete_speech_bytes)} bytes")
                    
                    # Reset for next speech
                    self.speech_frames = []
                    speech_start_time = None
                    
            except Exception as e:
                self.logger.error(f"Error in listening loop: {e}")
                break
    
    def stop_listening(self):
        """Stop listening"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        
        if self.recording_thread:
            self.recording_thread.join(timeout=2)
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            
        self.logger.info("Stopped listening")
    
    def save_audio_to_file(self, audio_data: bytes, filepath: str):
        """Save audio data to WAV file"""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with wave.open(str(filepath), 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(np.dtype(self.dtype).itemsize)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data)
                
            self.logger.info(f"Audio saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save audio: {e}")
            raise
    
    def get_audio_devices(self):
        """List available audio input devices"""
        try:
            devices = []
            device_list = sd.query_devices()
            
            for i, device in enumerate(device_list):
                if device['max_input_channels'] > 0:
                    devices.append({
                        'index': i,
                        'name': device['name'],
                        'channels': device['max_input_channels'],
                        'sample_rate': device['default_samplerate']
                    })
                    
            return devices
        except Exception as e:
            self.logger.error(f"Error querying audio devices: {e}")
            return []
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_listening()
        self.logger.info("Audio resources cleaned up")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()