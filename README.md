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

## 🚀 Installation & Usage

```bash
# Clone repository
git clone <your-repo-url>
cd AI-Video-Summarizer

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app.py
