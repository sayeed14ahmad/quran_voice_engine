#solved the 101 issue by adding the missing import statements and ensuring the \
# model files exist before starting the application.


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
        print(f"\n--- Processing started ---")
        if not audio_path:
            return "Please use the microphone to record a verse."
        
        try:
            # Read and format audio
            wave, sr = sf.read(audio_path, dtype="float32", always_2d=True)
            print(f"Original audio shape: {wave.shape}, Sample rate: {sr}")
            
            wave = wave.mean(axis=1) 
            if sr != 16000:
                g = np.gcd(sr, 16000)
                wave = resample_poly(wave, 16000 // g, sr // g).astype(np.float32)
                print(f"Resampled audio shape: {wave.shape}")
                
            # FORCE array to be contiguous in memory (crucial for C++ models)
            wave = np.ascontiguousarray(wave)
            print("Audio formatted successfully. Passing to AI model...")
                
            # Run the AI transcription
            stream = recognizer.create_stream()
            stream.accept_waveform(16000, wave)
            recognizer.decode_stream(stream)
            
            # Check raw output
            raw_text = stream.result.text
            print(f"Raw model output: '{raw_text}'")
            
            # Clean up and format
            # Using multiple replace variations to catch different SentencePiece spacer encodings
            raw_words = raw_text.replace("\u2581", " ").split()
            final_text = " ".join(spelling_table.get(w, w) for w in raw_words)
            
            print(f"Final cleaned output: '{final_text}'")
            return final_text
            
        except Exception as e:
            error_msg = f"ERROR during processing: {str(e)}"
            print(error_msg)
            return error_msg


# Build the user interface
iface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(sources=["microphone"], type="filepath", label="Click Record and Recite"),
    outputs=gr.Textbox(label="Recognized Quranic Arabic", rtl=True, text_align="right"),
    title="Quran Voice Recognition",
    description="Speak into the microphone. Processes 100% offline using quran-asr-small."
)

iface.launch()