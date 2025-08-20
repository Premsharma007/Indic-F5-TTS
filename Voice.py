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
import os
from transformers import AutoModel

warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

# Function to load reference audio from file
def load_ref_audio(path):
    if os.path.exists(path):
        audio_data, sample_rate = sf.read(path)
        return sample_rate, audio_data
    return None, None

# Function to load reference text
def load_ref_text(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

@spaces.GPU
def synthesize_speech(text, ref_audio, ref_text):
    start_time = time.time()

    if ref_audio is None or ref_text.strip() == "":
        total_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        return (24000, np.zeros(1, dtype=np.float32)), total_time_str

    if isinstance(ref_audio, tuple) and len(ref_audio) == 2:
        sample_rate, audio_data = ref_audio
    else:
        total_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        return (24000, np.zeros(1, dtype=np.float32)), total_time_str

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        sf.write(temp_audio.name, audio_data, samplerate=sample_rate, format='WAV')
        temp_audio.flush()

    try:
        audio = model(text, ref_audio_path=temp_audio.name, ref_text=ref_text)
    except Exception as e:
        print(f"Error generating audio: {e}")
        total_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        return (24000, np.zeros(1, dtype=np.float32)), total_time_str

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    total_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    return (24000, audio), total_time_str


# Load TTS model
repo_id = "6Morpheus6/IndicF5"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device", device)
model = model.to(device)


# Gradio interface
with gr.Blocks() as iface:
    gr.Markdown(
        """
        # **Spider『X』(T2S) Indic_F5 Speech Model - 1.0** 
        
        Generate speech using a reference prompt audio and its corresponding text.
        """
    )

    with gr.Row():
        with gr.Column():
            # Toggles
            txt_input_toggle = gr.Checkbox(label="Use .txt file for Input", value=False)
            ref_txt_toggle = gr.Checkbox(label="Use .txt file for Reference Text", value=False)
            auto_ref_toggle = gr.Checkbox(label="Auto Load Reference", value=False)

            # Input text / file
            text_input = gr.Textbox(label="Text to Synthesize", placeholder="Enter text here...", lines=3, visible=True)
            text_file_input = gr.File(label="Upload Text File", file_types=[".txt"], visible=False)

            # Reference audio/text (manual)
            ref_audio_input = gr.Audio(type="numpy", label="Reference Prompt Audio", visible=True)
            ref_text_input = gr.Textbox(label="Reference Text", placeholder="Enter transcript...", lines=2, visible=True)

            # Reference text file upload
            ref_text_file = gr.File(label="Upload Reference Text File", file_types=[".txt"], visible=False)

            submit_btn = gr.Button("🎤 Generate Speech", variant="primary")

        with gr.Column():
            output_audio = gr.Audio(label="Generated Speech", type="numpy")
            time_taken = gr.Label(label="Total Time Taken (HH:MM:SS)")

    # --- Toggle Logic ---
    def toggle_inputs(txt_toggle, ref_toggle, auto_ref):
        updates = {}

        # Input text vs file
        updates["text_input"] = gr.update(visible=not txt_toggle)
        updates["text_file_input"] = gr.update(visible=txt_toggle)

        # Reference manual vs file
        updates["ref_text_input"] = gr.update(visible=not ref_toggle)
        updates["ref_text_file"] = gr.update(visible=ref_toggle)

        # Auto-load reference
        if auto_ref:
            ref_audio_path = os.path.join("Reference_Text_Audio", "Reference_Audio.mp3")
            ref_text_path = os.path.join("Reference_Text_Audio", "Reference_Text.txt")
            sr, audio_data = load_ref_audio(ref_audio_path)
            ref_text_val = load_ref_text(ref_text_path)
            if sr and audio_data is not None:
                updates["ref_audio_input"] = gr.update(value=(sr, audio_data), visible=True)
            else:
                updates["ref_audio_input"] = gr.update(visible=True)
            if ref_text_val:
                updates["ref_text_input"] = gr.update(value=ref_text_val, visible=True)
            else:
                updates["ref_text_input"] = gr.update(visible=True)
        else:
            updates["ref_audio_input"] = gr.update(visible=True)
            updates["ref_text_input"] = gr.update(visible=not ref_toggle)
            updates["ref_text_file"] = gr.update(visible=ref_toggle)

        return updates

    txt_input_toggle.change(
        toggle_inputs,
        inputs=[txt_input_toggle, ref_txt_toggle, auto_ref_toggle],
        outputs={
            "text_input": text_input,
            "text_file_input": text_file_input,
            "ref_text_input": ref_text_input,
            "ref_text_file": ref_text_file,
            "ref_audio_input": ref_audio_input,
        },
    )

    ref_txt_toggle.change(
        toggle_inputs,
        inputs=[txt_input_toggle, ref_txt_toggle, auto_ref_toggle],
        outputs={
            "text_input": text_input,
            "text_file_input": text_file_input,
            "ref_text_input": ref_text_input,
            "ref_text_file": ref_text_file,
            "ref_audio_input": ref_audio_input,
        },
    )

    auto_ref_toggle.change(
        toggle_inputs,
        inputs=[txt_input_toggle, ref_txt_toggle, auto_ref_toggle],
        outputs={
            "text_input": text_input,
            "text_file_input": text_file_input,
            "ref_text_input": ref_text_input,
            "ref_text_file": ref_text_file,
            "ref_audio_input": ref_audio_input,
        },
    )

    # --- Submit logic ---
    def prepare_inputs(text, text_file, ref_audio, ref_text, ref_text_file):
        if text_file is not None:
            with open(text_file.name, "r", encoding="utf-8") as f:
                text = f.read().strip()
        if ref_text_file is not None:
            with open(ref_text_file.name, "r", encoding="utf-8") as f:
                ref_text = f.read().strip()
        return text, ref_audio, ref_text

    def run_pipeline(text, text_file, ref_audio, ref_text, ref_text_file):
        text, ref_audio, ref_text = prepare_inputs(text, text_file, ref_audio, ref_text, ref_text_file)
        return synthesize_speech(text, ref_audio, ref_text)

    submit_btn.click(
        run_pipeline,
        inputs=[text_input, text_file_input, ref_audio_input, ref_text_input, ref_text_file],
        outputs=[output_audio, time_taken],
    )

iface.launch()
