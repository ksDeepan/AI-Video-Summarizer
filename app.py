import os
import sys
import shutil
import subprocess
from pathlib import Path
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import moviepy.editor as mp
import imageio_ffmpeg
from utils import get_whisper_model, get_summarizer


# === ✅ Setup ffmpeg ===
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
os.environ["PATH"] = str(Path(ffmpeg_exe).parent) + os.pathsep + os.environ.get("PATH", "")

print("🔍 Checking ffmpeg path...")
if shutil.which("ffmpeg"):
    print(f"✅ ffmpeg found at: {shutil.which('ffmpeg')}")
else:
    print("❌ ffmpeg not found. Aborting.")
    sys.exit(1)

try:
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
    print("✅ ffmpeg is working in Python!")
    print("➡️  ffmpeg version:", result.stdout.splitlines()[0])
except Exception as e:
    print(f"❌ ffmpeg failed to run: {e}")
    sys.exit(1)


# === Paths ===
INPUT_VIDEO = "inputs/sample.mp4"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

AUDIO_FILE = OUTPUT_DIR / "temp_audio.wav"
TRANSCRIPT_FILE = OUTPUT_DIR / "transcript.txt"
SUMMARY_FILE = OUTPUT_DIR / "summary.txt"
PDF_FILE = OUTPUT_DIR / "video_summary_report.pdf"


# === 🔊 Audio Extraction ===
def extract_audio(video_path, audio_path):
    print(f"🎧 Extracting audio from: {video_path}")
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)
    try:
        video = mp.VideoFileClip(str(video_path))
        if video.audio is None:
            print("❌ No audio track found in video.")
            sys.exit(1)
        video.audio.write_audiofile(str(audio_path))
        video.close()
    except Exception as e:
        print(f"❌ Audio extraction failed: {e}")
        sys.exit(1)
    return audio_path


# === 🧾 PDF Generation ===
def generate_pdf(transcript, summary, pdf_path):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    x, y = 50, height - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, "🎬 Video Summary Report")
    y -= 40

    # Transcript Section
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Transcript:")
    y -= 20
    c.setFont("Helvetica", 10)

    for line in transcript.splitlines():
        for sub in [line[i:i+100] for i in range(0, len(line), 100)]:
            if y < 50:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 10)
            c.drawString(x, y, sub)
            y -= 15

    if y < 100:
        c.showPage()
        y = height - 50

    # Summary Section
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Summary:")
    y -= 20
    c.setFont("Helvetica", 10)

    for line in summary.splitlines():
        for sub in [line[i:i+100] for i in range(0, len(line), 100)]:
            if y < 50:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 10)
            c.drawString(x, y, sub)
            y -= 15

    c.save()
    buffer.seek(0)

    with open(pdf_path, "wb") as f:
        f.write(buffer.read())


# === 🚀 Main Pipeline ===
def main():
    print("🎬 Starting video summarization pipeline...")

    # 1. Extract audio
    audio_file = extract_audio(INPUT_VIDEO, AUDIO_FILE)
    if not audio_file.exists():
        print(f"❌ Audio file not created: {audio_file}")
        sys.exit(1)

    # 2. Transcribe using Whisper
    print("📝 Transcribing audio...")
    whisper_model = get_whisper_model()
    try:
        result = whisper_model.transcribe(str(audio_file))
        transcript = result["text"]
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        sys.exit(1)

    with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
        f.write(transcript)

    # 3. Summarize (with chunking for long transcripts)
    print("🧠 Generating summary...")
    summarizer = get_summarizer()

    def chunk_text(text, max_tokens=800):
        words = text.split()
        for i in range(0, len(words), max_tokens):
            yield " ".join(words[i:i+max_tokens])

    try:
        chunks = list(chunk_text(transcript, max_tokens=800))
        summaries = []
        for i, chunk in enumerate(chunks, 1):
            print(f"🔹 Summarizing chunk {i}/{len(chunks)}...")
            part_summary = summarizer(
                chunk,
                max_length=100,
                min_length=30,
                do_sample=False
            )[0]["summary_text"]
            summaries.append(part_summary)

        summary = " ".join(summaries)
    except Exception as e:
        print(f"❌ Summarization failed: {e}")
        sys.exit(1)

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(summary)

    # 4. Generate PDF
    print("📄 Creating PDF report...")
    try:
        generate_pdf(transcript, summary, PDF_FILE)
        print(f"✅ PDF saved at: {PDF_FILE}")
    except Exception as e:
        print(f"❌ Failed to create PDF: {e}")
        sys.exit(1)

    print("\n✅ All done!")
    print(f"📜 Transcript: {TRANSCRIPT_FILE}")
    print(f"📝 Summary:   {SUMMARY_FILE}")
    print(f"📄 PDF Report:{PDF_FILE}")


if __name__ == "__main__":
    main()
