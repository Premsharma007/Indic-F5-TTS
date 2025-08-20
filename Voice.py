import io
import spaces
import torch
import librosa
import warnings
import tempfile
import numpy as np
import gradio as gr
import soundfile as sf
import time
import os
from transformers import AutoModel

# ------------------------------------------------ #
#   Suppress Warnings                              #
# ------------------------------------------------ #
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")


# ------------------------------------------------ #
#   Speech Synthesis Logic                         #
# ------------------------------------------------ #
@spaces.GPU
def synthesize_speech(text, ref_audio, ref_text, input_file_name=None):
    start_time = time.time()

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

    # Save ref audio temp
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        sf.write(temp_audio.name, audio_data, samplerate=sample_rate, format='WAV')
        temp_audio.flush()

    # Generate speech
    try:
        audio = model(text, ref_audio_path=temp_audio.name, ref_text=ref_text)
    except Exception as e:
        print(f"Error generating audio: {e}")
        total_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        return (24000, np.zeros(1, dtype=np.float32)), total_time_str

    # Normalize audio
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    # ------------------------------------------------ #
    #   Save Output Logic                              #
    # ------------------------------------------------ #
    os.makedirs("Outputs", exist_ok=True)

    if input_file_name:
        base_name = os.path.splitext(os.path.basename(input_file_name))[0]
        out_path = f"Outputs/TTS+{base_name}.wav"
        sf.write(out_path, audio, samplerate=24000)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_base = f"Outputs/TTS+{timestamp}"
        sf.write(out_base + ".wav", audio, samplerate=24000)
        with open(out_base + "_InpTxt.txt", "w", encoding="utf-8") as f:
            f.write(text)

    total_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    return (24000, audio), total_time_str


# ------------------------------------------------ #
#   Load TTS Model                                 #
# ------------------------------------------------ #
repo_id = "6Morpheus6/IndicF5"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device", device)
model = model.to(device)


# ------------------------------------------------ #
#   Gradio UI                                      #
# ------------------------------------------------ #
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="cyan",
        secondary_hue="blue",
        neutral_hue="gray",
        font=["Inter", "sans-serif"]
    ).set(
        background_fill="linear-gradient(135deg, #0f172a, #1e293b)",
        block_background_fill="#111827",
        block_title_text_color="white",
        block_border_color="#334155",
        body_text_color="#d1d5db",
        button_primary_background_fill="#0ea5e9",
        button_primary_text_color="white",
        button_secondary_background_fill="#1e293b",
        button_secondary_text_color="#e5e7eb"
    )
) as iface:

    gr.Markdown(
        """
        <div style="text-align: center; padding: 20px; border-radius: 16px; background: #0f172a; box-shadow: 0px 4px 12px rgba(0,0,0,0.6);">
            <h1 style="color:#38bdf8; margin-bottom: 8px;">Spider『X』(T2S) Indic_F5 Speech Model - 1.0</h1>
            <p style="color:#94a3b8;">Generate speech using a reference prompt audio and its corresponding text.</p>
        </div>
        """
    )

    # ---------------- Toggles ---------------- #
    with gr.Group():
        with gr.Row():
            txt_input_toggle = gr.Checkbox(label="Use .txt file for Input Text", value=False)
            ref_txt_toggle = gr.Checkbox(label="Use .txt file for Reference Text", value=False)
            reference_load_toggle = gr.Checkbox(label="Auto-load Reference from Folder", value=False)

    with gr.Row():
        with gr.Column():
            with gr.Group():
                text_input = gr.Textbox(label="Text to Synthesize", placeholder="Enter text...", lines=3, visible=True)
                text_file_input = gr.File(label="Upload Input Text File", type="file", visible=False)

            with gr.Group():
                ref_audio_input = gr.Audio(type="numpy", label="Reference Prompt Audio")
                ref_text_input = gr.Textbox(label="Text in Reference Prompt Audio", placeholder="Enter transcript...", lines=2, visible=True)
                ref_text_file = gr.File(label="Upload Reference Text File", type="file", visible=False)

            with gr.Group():
                submit_btn = gr.Button("🎤 Generate Speech", variant="primary", elem_classes="shadow-btn")

        with gr.Column():
            with gr.Group():
                output_audio = gr.Audio(label="Generated Speech", type="numpy")
                time_taken = gr.Label(label="Total Time Taken (HH:MM:SS)")

    # ---------------- Toggle Logic ---------------- #
    def toggle_inputs(use_txt_file, use_ref_file, use_ref_load):
        text_visible = not use_txt_file
        text_file_visible = use_txt_file
        ref_text_visible = not use_ref_file
        ref_text_file_visible = use_ref_file
        ref_audio_val, ref_text_val = None, None

        if use_ref_load:
            audio_path = "Reference_Text_Audio/Reference_Audio.mp3"
            text_path = "Reference_Text_Audio/Reference_Text.txt"
            if os.path.exists(audio_path):
                ref_audio_val = audio_path
            if os.path.exists(text_path):
                with open(text_path, "r", encoding="utf-8") as f:
                    ref_text_val = f.read()
        return (
            gr.update(visible=text_visible),
            gr.update(visible=text_file_visible),
            gr.update(visible=ref_text_visible, value=ref_text_val),
            gr.update(visible=ref_text_file_visible),
            ref_audio_val
        )

    txt_input_toggle.change(
        toggle_inputs,
        inputs=[txt_input_toggle, ref_txt_toggle, reference_load_toggle],
        outputs=[text_input, text_file_input, ref_text_input, ref_text_file, ref_audio_input]
    )
    ref_txt_toggle.change(
        toggle_inputs,
        inputs=[txt_input_toggle, ref_txt_toggle, reference_load_toggle],
        outputs=[text_input, text_file_input, ref_text_input, ref_text_file, ref_audio_input]
    )
    reference_load_toggle.change(
        toggle_inputs,
        inputs=[txt_input_toggle, ref_txt_toggle, reference_load_toggle],
        outputs=[text_input, text_file_input, ref_text_input, ref_text_file, ref_audio_input]
    )

    # ---------------- Submit Logic ---------------- #
    def process_inputs(text, text_file, ref_audio, ref_text, ref_text_file, input_file_name):
        if text_file is not None:
            input_file_name = text_file.name
            with open(text_file.name, "r", encoding="utf-8") as f:
                text = f.read()

        if ref_text_file is not None:
            with open(ref_text_file.name, "r", encoding="utf-8") as f:
                ref_text = f.read()

        return synthesize_speech(text, ref_audio, ref_text, input_file_name)

    submit_btn.click(
        process_inputs,
        inputs=[text_input, text_file_input, ref_audio_input, ref_text_input, ref_text_file, text_file_input],
        outputs=[output_audio, time_taken]
    )

# ------------------------------------------------ #
#   Launch Interface                               #
# ------------------------------------------------ #
iface.launch()
