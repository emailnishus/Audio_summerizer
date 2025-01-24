import whisper
import openai
from pathlib import Path
import logging
from typing import Optional
from dotenv import load_dotenv
import os

import warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning
)

import warnings
warnings.filterwarnings(
    "ignore",
    message="FP16 is not supported on CPU; using FP32 instead",
    category=UserWarning
)



class AudioSummarizer:
    def __init__(self, openai_api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize the AudioSummarizer with OpenAI credentials and models.

        Args:
            openai_api_key: Your OpenAI API key
            model: The GPT model to use for summarization (default: "gpt-3.5-turbo")
        """
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        self.model = model
        self.whisper_model = whisper.load_model("base")
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("AudioSummarizer")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def transcribe_audio(self, audio_path: str) -> Optional[str]:
        """
        Transcribe audio file using Whisper.

        Args:
            audio_path: Path to the audio file

        Returns:
            Transcribed text or None if transcription fails
        """
        try:
            self.logger.info(f"Starting transcription of {audio_path}")
            audio_path = Path(audio_path)

            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            result = self.whisper_model.transcribe(str(audio_path))
            transcribed_text = result["text"]

            self.logger.info("Transcription completed successfully")
            return transcribed_text

        except Exception as e:
            self.logger.error(f"Error during transcription: {str(e)}")
            return None

    def summarize_text(self, text: str, max_length: int = 250) -> Optional[str]:
        """
        Summarize text using OpenAI's GPT model.

        Args:
            text: Text to summarize
            max_length: Maximum length of the summary in words

        Returns:
            Summarized text or None if summarization fails
        """
        try:
            self.logger.info("Starting text summarization")

            prompt = f"""Please provide a clear and concise summary of the following text in no more than {max_length} words. 
            Focus on the main points and key takeaways:
            
            {text}"""

            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates clear and concise summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length * 2,  # Approximate token limit
                temperature=0.5
            )

            summary = response.choices[0].message.content
            self.logger.info("Summarization completed successfully")
            return summary

        except Exception as e:
            self.logger.error(f"Error during summarization: {str(e)}")
            return None

    def process_audio_file(self, audio_path: str, max_summary_length: int = 250) -> Optional[dict]:
        """
        Process an audio file by transcribing it and generating a summary.

        Args:
            audio_path: Path to the audio file
            max_summary_length: Maximum length of the summary in words

        Returns:
            Dictionary containing transcription and summary, or None if processing fails
        """
        # Transcribe audio
        transcription = self.transcribe_audio(audio_path)
        if not transcription:
            return None

        # Generate summary
        summary = self.summarize_text(transcription, max_summary_length)
        if not summary:
            return None

        return {
            "transcription": transcription,
            "summary": summary
        }

# Example usage
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Initialize summarizer
    summarizer = AudioSummarizer(
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Process audio file
    audio_file = r"C:\Users\email\Desktop\summerizer\english-poem-108554.mp3"
    result = summarizer.process_audio_file(audio_file)
    print(f"Audio file path: {audio_file}")


    if result:
        print("\nTranscription:")
        print(result["transcription"])
        print("\nSummary:")
        print(result["summary"])
