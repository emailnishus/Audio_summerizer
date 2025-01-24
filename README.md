# Audio_summerizer

This project is an Audio Summarizer that uses OpenAI's GPT-3.5-turbo and Whisper to transcribe and summarize audio files. It processes an audio file, transcribes the content into text, and then generates a concise summary of the transcription.

Features
Transcription: Converts audio files (e.g., .mp3, .wav) into text using OpenAI's Whisper.
Summarization: Uses OpenAI's GPT-3.5-turbo to generate a concise summary of the transcription.
Error Handling: Includes logging for debugging errors during transcription or summarization.
Prerequisites
Ensure the following tools and libraries are installed:

Python (version 3.8 or higher)
Required Libraries (see the installation section below):
whisper
openai
dotenv
torch
pathlib
logging
