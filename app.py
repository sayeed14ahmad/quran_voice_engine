import gradio as gr
import sherpa_onnx
import soundfile as sf
import numpy as np
import json
import os
import sqlite3
import requests
import re
from scipy.signal import resample_poly
from rapidfuzz import process, fuzz

# ==========================================
# 1. BULLETPROOF FILE PATHS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "model.int8.onnx")
TOKENS_FILE = os.path.join(BASE_DIR, "tokens.txt")
SPELLING_FILE = os.path.join(BASE_DIR, "spelling.json")
JSON_DB_FILE = os.path.join(BASE_DIR, "quran_db.json")
SQL_DB_FILE = os.path.join(BASE_DIR, "quran_offline.db")

# ==========================================
# 2. INITIALIZE VOICE ENGINE
# ==========================================
recognizer = sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
    model=MODEL_FILE,
    tokens=TOKENS_FILE,
    num_threads=4
)

spelling_table = {}
if os.path.exists(SPELLING_FILE):
    with open(SPELLING_FILE, encoding="utf-8") as f:
        spelling_table = json.load(f)

# ==========================================
# 3. SETUP OFFLINE SQLITE DATABASE
# ==========================================
def clean_arabic(text):
    """Removes Tashkeel (diacritics) for accurate matching."""
    return re.sub(r'[\u064B-\u065F\u0670\u06D6-\u06ED]', '', text)


def setup_sqlite_db():
    """Create the local search table with pre-cleaned Arabic text."""
    conn = sqlite3.connect(SQL_DB_FILE)
    c = conn.cursor()

    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='quran_verses'")
    if c.fetchone():
        columns = {row[1] for row in c.execute("PRAGMA table_info(quran_verses)")}
        if "clean_text" not in columns:
            c.execute("DROP TABLE quran_verses")

    c.execute("""
        CREATE TABLE IF NOT EXISTS quran_verses (
            surah INTEGER, 
            ayah INTEGER, 
            arabic_text TEXT,
            clean_text TEXT
        );
    """)

    c.execute("SELECT count(*) FROM quran_verses")
    if c.fetchone()[0] == 0 and os.path.exists(JSON_DB_FILE):
        with open(JSON_DB_FILE, "r", encoding="utf-8") as file:
            quran_data = json.load(file)
            for verse in quran_data:
                arabic_text = verse["arabic"]
                c.execute(
                    """
                    INSERT INTO quran_verses
                        (surah, ayah, arabic_text, clean_text)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        verse["surah"],
                        verse["ayah"],
                        arabic_text,
                        clean_arabic(arabic_text),
                    ),
                )
        conn.commit()
    conn.close()

setup_sqlite_db()

# ==========================================
# 4. SEARCH FUNCTIONS (Multi-Output UI Integration)
# ==========================================
def search_offline(arabic_text):
    """Find the best local matches and return separated UI elements."""
    conn = sqlite3.connect(SQL_DB_FILE)
    c = conn.cursor()
    c.execute("SELECT surah, ayah, arabic_text, clean_text FROM quran_verses")
    all_verses = c.fetchall()
    conn.close()

    clean_voice = clean_arabic(arabic_text)
    clean_db_list = [row[3] for row in all_verses]
    matches = process.extract(
        clean_voice,
        clean_db_list,
        limit=5,
        scorer=fuzz.partial_ratio,
    )

    if not matches or matches[0][1] < 60:
        return (
            arabic_text,
            "❌ No clear match found.\n\n"
            "Try reciting 5–10 words closer to the microphone.",
            "",
        )

    best_match_md = ""
    alternatives_md = "### Other possible matches\n"

    for i, (_, score, index) in enumerate(matches):
        if score >= 60:
            surah, ayah, exact_text, _ = all_verses[index]

            if i == 0:
                confidence = (
                    "High"
                    if score >= 85
                    else "Possible match"
                    if score >= 65
                    else "Weak match"
                )
                best_match_md = (
                    f"### Surah {surah} — Ayah {ayah}\n\n"
                    "<div dir='rtl' style='font-size: 28px; "
                    f"text-align: right; margin: 15px 0;'>{exact_text}</div>\n\n"
                    f"**Match confidence:** {confidence}\n\n"
                    f"**Similarity:** {round(score)}%"
                )
            else:
                alternatives_md += (
                    f"**Surah {surah}, Ayah {ayah}** — "
                    f"Similarity: {round(score)}%\n\n"
                    "<div dir='rtl' style='text-align: right;'>"
                    f"{exact_text}</div>\n\n---\n"
                )

    if alternatives_md == "### Other possible matches\n":
        alternatives_md = ""

    return arabic_text, best_match_md, alternatives_md


def search_online(arabic_text):
    """Search Quran.com API, returning separated UI elements."""
    words = arabic_text.split()
    search_query = " ".join(words[:4]) if len(words) > 4 else arabic_text
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }

    try:
        response = requests.get(
            "https://api.quran.com/api/v4/search",
            params={"q": search_query, "size": 3, "language": "ar"},
            headers=headers,
            timeout=10,
        )
        if response.status_code != 200:
            return arabic_text, (
                f"❌ API blocked or failed (HTTP {response.status_code})."
            ), ""

        data = response.json()
        search_data = data.get("search", {})
        if search_data.get("total_results", 0) == 0:
            return arabic_text, (
                "❌ No matches found online, even with fuzzy search."
            ), ""

        best_match_md = ""
        alternatives_md = "### Other possible matches\n"
        for i, match in enumerate(search_data.get("results", [])[:3]):
            verse_key = match.get("verse_key", "Unknown")
            clean_text = re.sub(r"<[^>]+>", "", match.get("text", ""))
            if i == 0:
                best_match_md = (
                    f"### Verse {verse_key}\n\n"
                    "<div dir='rtl' style='font-size: 28px; "
                    f"text-align: right; margin: 15px 0;'>{clean_text}</div>\n\n"
                    "**Source:** ONLINE API"
                )
            else:
                alternatives_md += (
                    f"**Verse {verse_key}**\n\n"
                    "<div dir='rtl' style='text-align: right;'>"
                    f"{clean_text}</div>\n\n---\n"
                )

        if alternatives_md == "### Other possible matches\n":
            alternatives_md = ""

        return arabic_text, best_match_md, alternatives_md
    except requests.RequestException as exc:
        return arabic_text, f"⚠️ Online search failed: {exc}", ""
    except (ValueError, KeyError, TypeError) as exc:
        return arabic_text, f"⚠️ Invalid response from Quran.com: {exc}", ""

# ==========================================
# 5. AUDIO PROCESSING BRIDGE
# ==========================================
def process_audio(audio_path, search_mode):
    """Process audio and return transcription, best match, and alternatives."""
    if not audio_path:
        return "", "⚠️ Please upload or record an audio file.", ""
    
    try:
        wave, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        wave = wave.mean(axis=1) 
        if sr != 16000:
            g = np.gcd(sr, 16000)
            wave = resample_poly(wave, 16000 // g, sr // g).astype(np.float32)
        wave = np.ascontiguousarray(wave)
            
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, wave)
        recognizer.decode_stream(stream)
        
        raw_words = stream.result.text.replace("▁", " ").replace(" ", " ").replace("\u2581", " ").split()
        final_arabic = " ".join(spelling_table.get(w, w) for w in raw_words)
        
        if not final_arabic.strip():
            return "", (
                "⚠️ No speech detected.\n\n"
                "Try reciting 5–10 words closer to the microphone."
            ), ""

        if "Online" in search_mode:
            return search_online(final_arabic)
        else:
            return search_offline(final_arabic)
            
    except Exception as e:
        return "", f"⚠️ ERROR: {str(e)}", ""

# ==========================================
# 6. USER INTERFACE
# ==========================================
with gr.Blocks(
    title="Quran Voice Finder",
    theme=gr.themes.Default(text_size=gr.themes.sizes.text_lg),
) as iface:
    gr.Markdown("# Quran Voice Finder")

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="🎙️ Recite a verse or upload audio",
            )

            with gr.Accordion("Advanced Options", open=False):
                search_toggle = gr.Radio(
                    choices=[
                        "Offline Database (Fuzzy Match)",
                        "Online API (Exact Match & Pages)",
                    ],
                    value="Offline Database (Fuzzy Match)",
                    label="Select Search Engine",
                )

            with gr.Row():
                submit_btn = gr.Button("🔍 Find Verse", variant="primary")
                clear_btn = gr.Button("↻ Try Again", variant="secondary")

        with gr.Column(scale=1):
            transcription_box = gr.Textbox(
                label="What was heard",
                interactive=False,
                rtl=True,
                text_align="right",
            )

            gr.Markdown("---")
            best_match_box = gr.Markdown(
                "### Best Match\nResults will appear here."
            )

            gr.Markdown("---")
            alternatives_box = gr.Markdown("")

    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, search_toggle],
        outputs=[transcription_box, best_match_box, alternatives_box],
    )

    clear_btn.click(
        fn=lambda: (None, "", "### Best Match\nResults will appear here.", ""),
        inputs=[],
        outputs=[
            audio_input,
            transcription_box,
            best_match_box,
            alternatives_box,
        ],
    )

if __name__ == "__main__":
    iface.launch()