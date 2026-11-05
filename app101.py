import gradio as gr
import sherpa_onnx
import soundfile as sf
import numpy as np
import json
import os
from scipy.signal import resample_poly

# Ensure model files exist before starting
required_files = ["model.int8.onnx", "tokens.txt"]
for f in required_files:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing {f}. Please download it from Hugging Face into this folder.")

# Initialize the offline voice recognizer
recognizer = sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
    model="model.int8.onnx",
    tokens="tokens.txt",
    num_threads=4
)

# Load spelling dictionary to fix text formatting
spelling_table = {}
if os.path.exists("spelling.json"):
    with open("spelling.json", encoding="utf-8") as f:
        spelling_table = json.load(f)

def process_audio(audio_path):
    if not audio_path:
        return "Please use the microphone to record a verse."
    
    # Read and format audio for the model (requires 16kHz mono)
    wave, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    wave = wave.mean(axis=1) 
    if sr != 16000:
        g = np.gcd(sr, 16000)
        wave = resample_poly(wave, 16000 // g, sr // g).astype(np.float32)
        
    # Run the AI transcription
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, wave)
    recognizer.decode_stream(stream)
    
    # Clean up and format the final Arabic text
    raw_words = stream.result.text.replace("▁", " ").split()
    final_text = " ".join(spelling_table.get(w, w) for w in raw_words)
    
    return final_text

# Build the user interface
iface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(sources=["microphone"], type="filepath", label="Click Record and Recite"),
    outputs=gr.Textbox(label="Recognized Quranic Arabic", rtl=True, text_align="right"),
    title="Quran Voice Recognition",
    description="Speak into the microphone. Processes 100% offline using quran-asr-small."
)

iface.launch()