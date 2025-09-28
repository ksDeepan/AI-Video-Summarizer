# 🎬 AI Video Summarizer  
> 📹 Extract ➝ 📝 Transcribe ➝ 🧠 Summarize ➝ 📄 Export  

AI-powered tool that **summarizes long videos into concise reports**.  
It uses **OpenAI Whisper** for transcription and **Hugging Face Transformers** for summarization.  
Results can be exported as **PDF, transcript, or summary text files**.  

---

## ✨ Features
- 📤 Upload video files (`.mp4`, `.mov`, `.avi`)  
- 🎧 Automatic audio extraction  
- 📝 Whisper ASR (speech → text)  
- 🧠 Abstractive summarization with Transformers  
- 📄 Export as PDF or TXT  
- ⬇️ Download transcript & summary  

---

## 🛠️ Tech Stack & 📜 Workflow

```text
🛠️ Tech Stack
--------------
- Streamlit – UI framework
- MoviePy – video/audio processing
- OpenAI Whisper – transcription
- Transformers – summarization
- ReportLab – PDF generation


📜 Workflow
-----------
1. Upload a video
2. Audio is extracted automatically
3. Transcript is generated with Whisper
4. Summary is created with Transformers
5. Download Transcript, Summary, and PDF report
