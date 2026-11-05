# Quran Voice Engine v1

Local Quranic recitation recognition and verse search for Windows.

The application accepts a microphone recording or an uploaded audio file,
transcribes it locally with `sherpa-onnx`, and searches a bundled Quran
database for likely verse matches. The primary workflow is offline after the
project files and Python dependencies have been installed.

## What v1 does

- Captures audio from a microphone or accepts an uploaded audio file.
- Converts audio to mono, resamples it to 16 kHz, and sends it to the local
  ONNX speech-recognition model.
- Produces Arabic transcription using the bundled token and spelling files.
- Searches all 6,236 local Quran verses with RapidFuzz.
- Returns up to five local candidate matches with Surah, Ayah, score, and
  Arabic text.
- Creates or updates the local SQLite search database automatically.
- Includes an optional experimental Quran.com search mode that requires an
  internet connection.

## Current limitations

- This is a fuzzy candidate finder, not a religious authority or a guaranteed
  verse-identification system.
- Recognition quality depends on microphone quality, recitation clarity,
  background noise, recording length, and the bundled ASR model.
- Short or common phrases can produce several plausible candidates.
- v1 does not include Indo-Pak 15-line Hafizi page mapping.
- Online mode is experimental and may fail because of network restrictions,
  API availability, or changes to the external response format.
- The default and recommended mode is **Offline Database (Fuzzy Match)**.

## Project layout

```text
quran_voice_engine/
├── app.py                 # Main Gradio application
├── app101.py              # Historical early version
├── app102.py              # Historical transcription/search version
├── appv3.py               # Historical development version
├── model.int8.onnx        # Bundled sherpa-onnx ASR model
├── tokens.txt             # Model token vocabulary
├── spelling.json          # ASR spelling corrections
├── quran_db.json          # Local Quran verse source data
├── quran_offline.db       # Bundled generated SQLite search database
├── download_quran.py      # Optional data-download utility
├── requirements.txt       # Pinned Python dependencies
├── LICENSE                # MIT license for project source code
└── README.md              # This documentation
```

The `.venv`, Python cache files, Gradio cache, and temporary audio files are
intentionally excluded from Git.

## Requirements

- Windows 10 or Windows 11
- Python 3.13
- A working microphone for live recording
- Approximately 1 GB of free disk space for Python dependencies and local
  assets
- A modern browser such as Chrome, Edge, or Firefox

The application itself is designed to run locally. Internet is not required
for offline transcription/search after installation, but it may be needed to
install packages or use the optional online mode.

## Installation on Windows

Open PowerShell and run:

```powershell
git clone https://github.com/YOUR_USERNAME/quran_voice_engine.git
cd quran_voice_engine

py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the project interpreter directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `py -3.13` is unavailable, install Python 3.13 from the official Python
website and enable the Python launcher during installation.

## Run the application

With the virtual environment activated:

```powershell
python app.py
```

Or, without activating it:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open the local address printed in the terminal. It is normally:

```text
http://127.0.0.1:7860
```

Keep the terminal open while using the web interface. Stop the application
with `Ctrl+C`.

## Use offline mode

1. Open the local Gradio address.
2. Record a recitation with **Recite a Verse**, or upload an audio file.
3. Select **Offline Database (Fuzzy Match)**.
4. Click **Transcribe & Search**.
5. Review the recognized Arabic text and the candidate Surah/Ayah results.

For best results:

- Recite clearly and close to the microphone.
- Avoid music, fans, traffic, and other background noise.
- Include several consecutive words instead of a single short word.
- Avoid clipping the beginning or end of the recording.
- Use a normal speaking volume and do not place the microphone directly
  against the mouth.

## Test the local search without recording audio

This verifies the local SQLite database and RapidFuzz search independently of
the microphone and ASR model:

```powershell
$env:PYTHONIOENCODING="utf-8"
.\.venv\Scripts\python.exe -c "import app; print(app.search_offline('الحمد لله رب العالمين'))"
```

The output should include a strong candidate for **Surah 1, Ayah 2**.

If the Windows console cannot display Arabic, the `PYTHONIOENCODING` setting
above enables UTF-8 output for that command.

## How the offline pipeline works

```text
Microphone/upload
        ↓
soundfile reads audio
        ↓
Stereo-to-mono conversion
        ↓
scipy resample_poly → 16,000 Hz
        ↓
sherpa-onnx local ONNX recognizer
        ↓
spelling corrections
        ↓
Arabic normalization for comparison
        ↓
RapidFuzz partial matching
        ↓
Top local Surah/Ayah candidates
```

On startup, `app.py` checks `quran_offline.db`. If the search table is missing
or has the older schema, it rebuilds the generated table from `quran_db.json`
and stores pre-cleaned Arabic text for faster searches.

## Optional online mode

The interface includes an experimental online option backed by Quran.com.
Select **Online API** only when an internet connection is available.

Online mode:

- Sends a shortened Arabic query to an external service.
- Is not required for local transcription or offline matching.
- Can return HTTP errors, empty responses, or unavailable results.
- May stop working if the external API changes.

No cloud speech-recognition service is used by the local voice engine.

## Troubleshooting

### `ModuleNotFoundError`

Use the virtual-environment interpreter rather than a different system
Python:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

### Model or token file not found

Run the command from the project directory and confirm these files exist:

```powershell
Get-ChildItem model.int8.onnx, tokens.txt, spelling.json, quran_db.json
```

The application uses paths relative to `app.py`, so the files must remain in
the project root unless the code is changed.

### Microphone does not work

- Allow microphone access for the browser in Windows privacy settings.
- Allow microphone access when the browser prompts.
- Try uploading a WAV file to separate browser permissions from audio
  processing.
- Close other applications that may exclusively hold the microphone.

### No close offline match

- Record more words.
- Recite more slowly and clearly.
- Reduce background noise.
- Confirm that `quran_db.json` contains the complete corpus.
- Run the direct search test shown above.

### Database problems

`quran_offline.db` is generated from `quran_db.json`. Stop the application and
run it again after changing the source data. If a local database becomes
corrupted, make a backup and remove only that generated file; the next launch
will rebuild it.

### Gradio starts but the page is inaccessible

Confirm that the terminal shows a local URL and that no other application is
using the port. Try opening `http://127.0.0.1:7860` directly instead of using
the browser link.

## Data, model, and attribution

The MIT license in this repository applies only to the original project
source code. It does not automatically relicense bundled third-party assets.

The repository includes:

- A `sherpa-onnx` ONNX model associated with the
  `Abdou/quran-asr-small` model family.
- Quran Arabic data in `quran_db.json`.
- ASR token and spelling resources.

Before redistributing a fork or repackaged copy, review the upstream license
and usage terms for each asset. Preserve the relevant attribution and license
notices. The exact upstream terms should be confirmed from the source where
each asset was obtained.

`download_quran.py` is an optional helper that retrieves Quran data from the
Al Quran Cloud API. It is not used by the runtime when the bundled local data
is present, and it requires an internet connection.

## Privacy

Audio is processed by the local application and is not sent to a speech
recognition cloud service in offline mode. Temporary audio handling is
performed by the browser and Gradio runtime. Review your browser and operating
system microphone settings if privacy requirements are strict.

## Future work

- Improve ranking with combined Arabic similarity scores.
- Add confidence-aware result labels.
- Add validated page mapping for a specifically identified Indo-Pak
  15-line Hafizi edition.
- Add automated tests for known recitation samples.
- Improve noise handling and voice-activity detection.

## License

Original project source code is released under the MIT License. See
[LICENSE](LICENSE). Third-party assets remain subject to their own terms.
