# AI Video Summarizer

An AI-powered video summarization app using **Whisper** for transcription and **Transformers** for generating concise summaries.  
Users can upload a video, see the transcript and summary, and download a PDF report.

🎬 **Live Demo**: https://ai-video-summarizer-tgd4pxlmb22jrr3pyaqrp3.streamlit.app/

---

## Features

- Upload video files (MP4, MOV, AVI)  
- Extract audio from video  
- Transcribe audio using Whisper  
- Generate summaries using a transformer model  
- Create a downloadable PDF report (with transcript + summary)  
- Downloadable transcript and summary text files  

---

## Tech Stack & Dependencies

- Python  
- Streamlit (for frontend interface)  
- Whisper (OpenAI / `openai-whisper`) for transcription  
- Transformers / Sentence-Transformers (for summarization)  
- MoviePy (for handling video/audio extraction)  
- ReportLab (for PDF generation)  
- imageio-ffmpeg / ffmpeg (for audio/video processing)  


