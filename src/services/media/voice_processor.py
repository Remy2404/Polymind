import io
import os
import logging
import asyncio
import tempfile
from typing import Tuple, Optional, Union, Dict, Any
import speech_recognition as sr
from pydub import AudioSegment
import uuid


class VoiceProcessor:
    """
    Handles voice message processing and transcription.
    """

    def __init__(self):
        """Initialize the voice processor"""
        self.logger = logging.getLogger(__name__)
        self.temp_dir = tempfile.gettempdir()

    async def download_and_convert(
        self, voice_file, user_id: str, is_khmer: bool = False
    ) -> Tuple[str, str]:
        """
        Download voice file and convert to WAV format for speech recognition

        Args:
            voice_file: Telegram file object
            user_id: User ID for file naming
            is_khmer: Flag for Khmer language processing

        Returns:
            Tuple[str, str]: Path to original file and converted WAV file
        """
        try:
            # Generate a unique filename to avoid conflicts
            file_unique_id = str(uuid.uuid4())[:8]
            ogg_file_path = os.path.join(
                self.temp_dir, f"voice_{user_id}_{file_unique_id}.ogg"
            )
            wav_file_path = os.path.join(
                self.temp_dir, f"voice_{user_id}_{file_unique_id}.wav"
            )

            # Download the voice file
            await voice_file.download_to_drive(ogg_file_path)

            # Convert OGG to WAV
            await self._convert_to_wav(ogg_file_path, wav_file_path, is_khmer)

            return ogg_file_path, wav_file_path

        except Exception as e:
            self.logger.error(f"Error downloading/converting voice: {str(e)}")
            raise ValueError(f"Failed to process voice file: {str(e)}")

    async def _convert_to_wav(
        self, input_path: str, output_path: str, is_khmer: bool = False
    ) -> None:
        """
        Convert an audio file to WAV format optimized for speech recognition

        Args:
            input_path: Path to input audio file
            output_path: Path for output WAV file
            is_khmer: Flag for Khmer language processing
        """
        try:

            def _do_convert():
                # Load the audio file
                audio = AudioSegment.from_file(input_path)

                # Apply preprocessing for Khmer if needed
                if is_khmer:
                    # Boost high frequencies and apply noise reduction for better results with Khmer
                    audio = audio.high_pass_filter(300)  # Reduce low-frequency noise
                    audio = audio + 3  # Boost volume slightly

                # Export as WAV with speech recognition friendly settings
                audio.export(
                    output_path,
                    format="wav",
                    parameters=["-ar", "16000", "-ac", "1"],  # 16kHz, mono
                )

            # Run conversion in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _do_convert)

        except Exception as e:
            self.logger.error(f"Audio conversion error: {str(e)}")
            raise ValueError(f"Failed to convert audio: {str(e)}")

    async def transcribe(
        self, audio_file_path: str, language: str = "en-US", is_khmer: bool = False
    ) -> Tuple[str, str]:
        """
        Transcribe a voice file to text

        Args:
            audio_file_path: Path to WAV file
            language: Language code for recognition
            is_khmer: Flag for Khmer language processing

        Returns:
            Tuple[str, str]: Transcribed text and detected language
        """
        try:
            # Initialize recognizer
            recognizer = sr.Recognizer()

            # Function to perform transcription
            def _do_transcribe():
                with sr.AudioFile(audio_file_path) as source:
                    # Adjust for ambient noise and record
                    recognizer.adjust_for_ambient_noise(source)
                    audio = recognizer.record(source)

                    # Try Google's speech recognition
                    try:
                        text = recognizer.recognize_google(audio, language=language)
                        return text, language
                    except sr.UnknownValueError:
                        # If Khmer failed, try English as fallback
                        if is_khmer:
                            try:
                                text = recognizer.recognize_google(
                                    audio, language="en-US"
                                )
                                return text, "en-US"
                            except:
                                return "", language
                        return "", language
                    except Exception as e:
                        self.logger.error(f"Speech recognition error: {str(e)}")
                        return "", language

            # Run speech recognition in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _do_transcribe)

            # Clean up temp files in background
            asyncio.create_task(self._cleanup_files(audio_file_path))

            return result

        except Exception as e:
            self.logger.error(f"Transcription error: {str(e)}")
            # Attempt to clean up even on error
            asyncio.create_task(self._cleanup_files(audio_file_path))
            return "", language

    async def _cleanup_files(self, *file_paths) -> None:
        """
        Clean up temporary files

        Args:
            file_paths: Paths to files that should be deleted
        """
        for path in file_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    self.logger.warning(f"Failed to remove temp file {path}: {str(e)}")
