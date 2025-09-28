import whisper
from transformers import pipeline
import imageio_ffmpeg as ffmpeg
print("FFmpeg path:", ffmpeg.get_ffmpeg_exe())


# Cache models so they load only once
_whisper_model = None
_summarizer = None

def get_whisper_model(model_name="base"):
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model(model_name)
    return _whisper_model

def get_summarizer(model_name="sshleifer/distilbart-cnn-12-6"):
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline("summarization", model=model_name)
    return _summarizer
