"""Record the README demo GIF from the live app.

Run: uv run --with playwright python -m scripts.record_demo
Needs ffmpeg on PATH (or FFMPEG=path\\to\\ffmpeg.exe) and `playwright install chromium` once.
"""
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

# /~/+/ serves the app without Streamlit Cloud's frame and "Manage app" bar
URL = "https://coffee-assist.streamlit.app/~/+/"
QUESTIONS = [
    "What is crema on espresso?",            # answer from the coffee database
    "What does arabica coffee cost today?",  # current data, so web search with a link
    "Wie viel Koffein hat ein Espresso?",    # answers in the user's language
    "What's the weather in Berlin tomorrow?",  # off-topic, declined without web search
]
SIZE = {"width": 900, "height": 720}
OUT = Path(__file__).resolve().parent.parent / "images" / "demo.gif"
READ_PAUSE_MS = 3500  # time to read each answer before the next question


def record(video_dir):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport=SIZE, record_video_dir=video_dir, record_video_size=SIZE)
        start = time.monotonic()
        page = context.new_page()
        page.goto(URL, timeout=90_000)

        box = page.get_by_placeholder("Ask about coffee...")
        box.wait_for(timeout=90_000)
        page.wait_for_timeout(1000)
        ready = time.monotonic() - start  # the video before this is only the loading screen

        running = page.get_by_test_id("stStatusWidget")  # Streamlit's "Running… / Stop" corner
        for i, question in enumerate(QUESTIONS, start=1):
            # typing during a rerun gets wiped when Streamlit redraws the input box
            running.wait_for(state="hidden", timeout=90_000)
            box.click()
            page.keyboard.type(question, delay=55)
            if box.input_value() != question:
                raise RuntimeError(f"input box lost the question: {box.input_value()!r}")
            page.keyboard.press("Enter")
            # the answer is done when its message exists and the run has finished
            page.get_by_test_id("stChatMessage").nth(2 * i - 1).wait_for(timeout=90_000)
            page.wait_for_timeout(500)
            running.wait_for(state="hidden", timeout=90_000)
            page.wait_for_timeout(READ_PAUSE_MS)

        video = page.video.path()
        context.close()  # the video file is written on close
        browser.close()
    return video, ready


def to_gif(video, skip_seconds):
    ffmpeg = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    # palette built from the changing pixels + only the changed rectangle stored per frame:
    # a chat UI is mostly still, so this keeps the GIF small enough for a README
    # no dithering: the UI is flat colours, and dither noise changes pixels in every frame
    filters = ("fps=8,scale=760:-1:flags=lanczos,split[a][b];"
               "[a]palettegen=stats_mode=diff[p];"
               "[b][p]paletteuse=dither=none:diff_mode=rectangle")
    subprocess.run(
        [ffmpeg, "-y", "-ss", f"{skip_seconds:.1f}", "-i", str(video), "-vf", filters, "-loop", "0", str(OUT)],
        check=True, capture_output=True,
    )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        video, ready = record(tmp)
        to_gif(video, ready)
    print(f"saved {OUT} ({OUT.stat().st_size // 1024} KB)")
