import os
import shutil
import tempfile
from pathlib import Path
from io import BytesIO

import streamlit as st
import moviepy.editor as mp
import imageio_ffmpeg
from utils import get_whisper_model, get_summarizer

# 🛠️ PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# === ✅ Ensure ffmpeg is available ===
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
temp_ffmpeg_path = Path(tempfile.gettempdir()) / "ffmpeg.exe"
if not temp_ffmpeg_path.exists():
    shutil.copy(ffmpeg_exe, temp_ffmpeg_path)
os.environ["PATH"] = str(temp_ffmpeg_path.parent) + os.pathsep + os.environ.get("PATH", "")

# === 🗂️ Output directory ===
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# === 🎈 UI ===
st.title("🎬 AI Video Summarizer (Whisper + Transformers)")

uploaded_file = st.file_uploader("📤 Upload a video file", type=["mp4", "mov", "avi"])

# These will store results for PDF generation
transcript = ""
summary = ""

if uploaded_file:
    # ✅ Save uploaded file
    video_path = OUTPUT_DIR / uploaded_file.name
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ Uploaded video saved to: {video_path}")

    # 🎧 Extract audio
    audio_path = OUTPUT_DIR / "temp_audio.wav"
    st.write("🎧 Extracting audio...")
    try:
        video = mp.VideoFileClip(str(video_path))
        video.audio.write_audiofile(str(audio_path))
        st.success("✅ Audio extracted successfully.")
    except Exception as e:
        st.error(f"❌ Failed to extract audio: {e}")
        st.stop()

    # 📝 Transcribe
    st.write("📝 Transcribing using Whisper...")
    whisper_model = get_whisper_model()
    try:
        result = whisper_model.transcribe(str(audio_path))
        transcript = result["text"]
        with open(OUTPUT_DIR / "transcript.txt", "w", encoding="utf-8") as f:
            f.write(transcript)
        st.subheader("📜 Transcript")
        st.text_area("Transcript", transcript, height=250)
    except Exception as e:
        st.error(f"❌ Transcription failed: {e}")
        st.stop()

    # 🧠 Summarize
    st.write("🧠 Generating summary...")
    summarizer = get_summarizer()
    try:
        summary = summarizer(
            transcript,
            max_length=100,
            min_length=30,
            do_sample=False
        )[0]["summary_text"]
        with open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
            f.write(summary)
        st.subheader("📝 Summary")
        st.success(summary)
    except Exception as e:
        st.error(f"❌ Summarization failed: {e}")
        st.stop()

    # 🧾 PDF generation
    def generate_pdf(transcript, summary):
        buffer = BytesIO()
        p = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        x_margin = 50
        y = height - 50

        # Title
        p.setFont("Helvetica-Bold", 16)
        p.drawString(x_margin, y, "🎬 Video Summary Report")
        y -= 30

        # Transcript
        p.setFont("Helvetica-Bold", 12)
        p.drawString(x_margin, y, "Transcript:")
        y -= 20
        p.setFont("Helvetica", 10)
        for line in transcript.split("\n"):
            for subline in [line[i:i+100] for i in range(0, len(line), 100)]:
                if y < 50:
                    p.showPage()
                    y = height - 50
                    p.setFont("Helvetica", 10)
                p.drawString(x_margin, y, subline)
                y -= 15

        # Summary
        if y < 100:
            p.showPage()
            y = height - 50
        p.setFont("Helvetica-Bold", 12)
        p.drawString(x_margin, y, "Summary:")
        y -= 20
        p.setFont("Helvetica", 10)
        for line in summary.split("\n"):
            for subline in [line[i:i+100] for i in range(0, len(line), 100)]:
                if y < 50:
                    p.showPage()
                    y = height - 50
                    p.setFont("Helvetica", 10)
                p.drawString(x_margin, y, subline)
                y -= 15

        p.save()
        buffer.seek(0)
        return buffer

    # 📥 PDF download button
    pdf_bytes = generate_pdf(transcript, summary)
    st.subheader("📄 Download Full Report as PDF")
    st.download_button(
        label="📥 Download PDF",
        data=pdf_bytes,
        file_name="video_summary_report.pdf",
        mime="application/pdf"
    )

    # === ⬇️ Download buttons (AFTER processing) ===
    st.markdown("---")
    st.subheader("📥 Download your files")

    st.download_button(
        label="📜 Download Transcript (.txt)",
        data=transcript,
        file_name="transcript.txt",
        mime="text/plain"
    )

    st.download_button(
        label="📝 Download Summary (.txt)",
        data=summary,
        file_name="summary.txt",
        mime="text/plain"
    )
