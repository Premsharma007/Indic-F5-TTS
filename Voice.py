import io
import spaces
import torch
import librosa
import warnings
import requests
import tempfile
import numpy as np
import gradio as gr
import soundfile as sf
import time
import re
import os
from datetime import datetime
from transformers import AutoModel

warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

# Paths
REF_DIR = "Reference_Text_Audio"
OUTPUT_DIR = "Outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper: load reference audio & text from folder if available
def load_default_reference():
    ref_audio_path = os.path.join(REF_DIR, "Reference_Audio.mp3")
    ref_text_path = os.path.join(REF_DIR, "Reference_Text.txt")

    audio_tuple, text_str = None, ""
    if os.path.exists(ref_audio_path):
        audio_data, sr = librosa.load(ref_audio_path, sr=None, mono=True)
        audio_tuple = (sr, audio_data.astype(np.float32))
    if os.path.exists(ref_text_path):
        with open(ref_text_path, "r", encoding="utf-8") as f:
            text_str = f.read().strip()
    return audio_tuple, text_str

# Helper: split text into natural chunks
def split_into_chunks(text, max_chars=600):
    parts = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    chunks, buf, cur_len = [], [], 0
    for p in parts:
        if cur_len + len(p) + 1 > max_chars and buf:
            chunks.append(" ".join(buf))
            buf, cur_len = [p], len(p)
        else:
            buf.append(p)
            cur_len += len(p) + 1
    if buf:
        chunks.append(" ".join(buf))
    return chunks if chunks else [text]

@spaces.GPU
def synthesize_speech(text, ref_audio, ref_text, txt_file=None, ref_txt_file=None):
    start_time = time.time()

    # If TXT file uploaded for synthesis → override
    txt_filename = None
    if txt_file is not None:
        try:
            txt_filename = os.path.splitext(os.path.basename(txt_file.name))[0]
            with open(txt_file.name, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            pass

    # Auto-load default reference if user did not upload
    if ref_audio is None and ref_text.strip() == "":
        default_audio, default_text = load_default_reference()
        if default_audio is not None:
            ref_audio = default_audio
        if default_text:
            ref_text = default_text

    # If separate reference text file uploaded
    if ref_txt_file is not None:
        try:
            with open(ref_txt_file.name, "r", encoding="utf-8") as f:
                ref_text = f.read().strip()
        except Exception:
            pass

    # Handle missing inputs
    if ref_audio is None or ref_text.strip() == "":
        total_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        return (24000, np.zeros(1, dtype=np.float32)), total_time_str

    # Validate ref_audio format
    if isinstance(ref_audio, tuple) and len(ref_audio) == 2:
        sample_rate, audio_data = ref_audio
    else:
        total_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        return (24000, np.zeros(1, dtype=np.float32)), total_time_str

    # Save ref audio to temp
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        sf.write(temp_audio.name, audio_data, samplerate=sample_rate, format='WAV')
        temp_audio.flush()

    final_audio = []
    silence_gap = np.zeros(int(0.2 * 24000), dtype=np.float32)  # 200ms silence at 24kHz

    try:
        # Process text in chunks
        for chunk in split_into_chunks(text):
            audio = model(chunk, ref_audio_path=temp_audio.name, ref_text=ref_text)

            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            audio = np.asarray(audio, dtype=np.float32).flatten()

            final_audio.append(audio)
            final_audio.append(silence_gap)

        audio_out = np.concatenate(final_audio) if final_audio else np.zeros(1, dtype=np.float32)

    except Exception as e:
        print(f"Error generating audio: {e}")
        audio_out = np.zeros(1, dtype=np.float32)

    # Save output to Outputs/
    if txt_filename:  # If text file was uploaded
        out_name = f"TTS_{txt_filename}.wav"
    else:  # No text file → use timestamp
        out_name = f"TTS_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    sf.write(out_path, audio_out, 24000)
    print(f"✅ Output saved at: {out_path}")

    total_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    return (24000, audio_out), total_time_str


# Load TTS model
repo_id = "6Morpheus6/IndicF5"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device", device)
model = model.to(device)


# Define Gradio interface
with gr.Blocks() as iface:
    gr.Markdown(
        """
        # **Spider『X』(T2S) Indic_F5 Speech Model - 1.1** 
        
        Generate speech using a reference prompt audio and its corresponding text.  
        (If available, system auto-loads from `Reference_Text_Audio/`)
        """
    )
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Text to Synthesize", placeholder="Enter text or upload TXT file...", lines=3)
            txt_file_input = gr.File(label="Upload TXT File (optional)", file_types=[".txt"])
            ref_audio_input = gr.Audio(type="numpy", label="Reference Prompt Audio (optional)")
            ref_text_input = gr.Textbox(label="Reference Audio Transcript", placeholder="Enter transcript or auto-loads...", lines=2)
            ref_txt_file_input = gr.File(label="Upload Reference Text File (optional)", file_types=[".txt"])
            submit_btn = gr.Button("🎤 Generate Speech", variant="primary")
        
        with gr.Column():
            output_audio = gr.Audio(label="Generated Speech", type="numpy")
            time_taken = gr.Label(label="Total Time Taken (HH:MM:SS)") 

    theme=gr.themes.Soft()
    submit_btn.click(
        synthesize_speech,
        inputs=[text_input, ref_audio_input, ref_text_input, txt_file_input, ref_txt_file_input],
        outputs=[output_audio, time_taken] 
    )


iface.launch()
