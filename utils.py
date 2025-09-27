import whisper
from transformers import pipeline
import imageio_ffmpeg as ffmpeg

# Show ffmpeg path at startup (debugging)
print("✅ FFmpeg path:", ffmpeg.get_ffmpeg_exe())

# === Cache models so they load only once ===
_whisper_model = None
_summarizer = None

def get_whisper_model(model_name: str = "base"):
    """
    Load and cache the Whisper ASR model.
    Default = 'base' (lightweight for faster inference).
    """
    global _whisper_model
    if _whisper_model is None:
        print(f"🔍 Loading Whisper model: {model_name}")
        _whisper_model = whisper.load_model(model_name)
        print("✅ Whisper model loaded.")
    return _whisper_model


def get_summarizer(model_name: str = "sshleifer/distilbart-cnn-12-6"):
    """
    Load and cache the summarization pipeline.
    Default = distilbart-cnn-12-6 (smaller, faster, Streamlit-friendly).
    """
    global _summarizer
    if _summarizer is None:
        print(f"🔍 Loading summarizer model: {model_name}")
        _summarizer = pipeline("summarization", model=model_name)
        print("✅ Summarizer loaded.")
    return _summarizer
