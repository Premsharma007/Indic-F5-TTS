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
import shutil
from transformers import AutoModel

# ------------------------------------------------ #
#   Suppress Warnings                              #
# ------------------------------------------------ #
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

# ------------------------------------------------ #
#   Setup Directories                              #
# ------------------------------------------------ #
INPUTS_DIR = "Inputs"
OUTPUTS_DIR = "Outputs"
PROCESSED_DIR = "Processed"
REF_DIR = "Reference_Text_Audio"

for folder in [INPUTS_DIR, OUTPUTS_DIR, PROCESSED_DIR, REF_DIR]:
    os.makedirs(folder, exist_ok=True)

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

    # Validate ref_audio format (Gradio can pass filepath or numpy tuple)
    if isinstance(ref_audio, str):
        audio_data, sample_rate = librosa.load(ref_audio, sr=None)
    elif isinstance(ref_audio, tuple) and len(ref_audio) == 2:
        sample_rate, audio_data = ref_audio
    else:
        total_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        return (24000, np.zeros(1, dtype=np.float32)), total_time_str

    # Save ref audio temp
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        sf.write(temp_audio.name, audio_data, samplerate=sample_rate, format='WAV')
        temp_audio_path = temp_audio.name

    # Generate speech
    try:
        audio = model(text, ref_audio_path=temp_audio_path, ref_text=ref_text)
        
        # Convert torch tensor to numpy if necessary
        if torch.is_tensor(audio):
            audio = audio.cpu().numpy()
            
    except Exception as e:
        print(f"Error generating audio: {e}")
        if os.path.exists(temp_audio_path): os.remove(temp_audio_path)
        total_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        return (24000, np.zeros(1, dtype=np.float32)), total_time_str

    # Cleanup temp
    if os.path.exists(temp_audio_path): os.remove(temp_audio_path)

    # Normalize audio
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    # ------------------------------------------------ #
    #   Save Output Logic                              #
    # ------------------------------------------------ #
    if input_file_name:
        base_name = os.path.splitext(os.path.basename(input_file_name))[0]
        out_path = f"Outputs/{base_name}.wav"
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
#   Batch Processing Logic                         #
# ------------------------------------------------ #
def run_batch_process(file_list, ref_audio, ref_text, progress=gr.Progress()):
    if not file_list:
        return "No files selected."
    
    start_time = time.time()
    logs = []
    
    for i, filename in enumerate(file_list):
        progress(i / len(file_list), desc=f"Processing {filename}")
        input_path = os.path.join(INPUTS_DIR, filename)
        
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Call the existing synthesis function
            _, _ = synthesize_speech(content, ref_audio, ref_text, input_file_name=filename)
            
            # Move to Processed folder
            shutil.move(input_path, os.path.join(PROCESSED_DIR, filename))
            logs.append(f"✅ {filename} - Done")
        except Exception as e:
            logs.append(f"❌ {filename} - Error: {str(e)}")
            continue

    total_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    return f"Batch Finished in {total_time}\n\n" + "\n".join(logs)

# ------------------------------------------------ #
#   Helper Functions                               #
# ------------------------------------------------ #
def scan_inputs():
    files = [f for f in os.listdir(INPUTS_DIR) if f.endswith(".txt")]
    files.sort()
    return gr.update(choices=files, value=files)

def load_defaults():
    audio_path = os.path.join(REF_DIR, "Reference_Audio.mp3")
    text_path = os.path.join(REF_DIR, "Reference_Text.txt")
    ref_text = ""
    if os.path.exists(text_path):
        with open(text_path, "r", encoding="utf-8") as f:
            ref_text = f.read()
    return audio_path if os.path.exists(audio_path) else None, ref_text

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
with gr.Blocks() as iface:

    gr.Markdown(
        """
        <div style="text-align: center; padding: 20px; border-radius: 16px; background: #0f172a; box-shadow: 0px 4px 12px rgba(0,0,0,0.6);">
            <h1 style="color:#38bdf8; margin-bottom: 8px;">Spider『X』(T2S) Indic_F5 Speech Model - 1.0</h1>
            <p style="color:#94a3b8;">Generate speech using a reference prompt audio and its corresponding text.</p>
        </div>
        """
    )

    with gr.Tab("Single Synthesis"):
        with gr.Row():
            with gr.Column():
                txt_input_toggle = gr.Checkbox(label="Use .txt file for Input Text", value=False)
                text_input = gr.Textbox(label="Text to Synthesize", placeholder="Enter text...", lines=3)
                text_file_input = gr.File(label="Upload Input Text File", type="filepath", visible=False)
                
                ref_audio_input = gr.Audio(type="filepath", label="Reference Prompt Audio")
                ref_text_input = gr.Textbox(label="Text in Reference Prompt Audio", placeholder="Enter transcript...", lines=2)
                
                submit_btn = gr.Button("🎤 Generate Speech", variant="primary")

            with gr.Column():
                output_audio = gr.Audio(label="Generated Speech", type="numpy")
                time_taken = gr.Label(label="Total Time Taken (HH:MM:SS)")

    with gr.Tab("Batch Processing"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 1. Setup Reference")
                load_def_btn = gr.Button("📂 Load Defaults (Ref_Audio/Ref_Text)")
                batch_ref_audio = gr.Audio(type="filepath", label="Batch Reference Audio")
                batch_ref_text = gr.Textbox(label="Batch Reference Text", lines=2)
                
                gr.Markdown("### 2. Queue")
                scan_btn = gr.Button("🔍 Scan Inputs/ Folder")
                file_queue = gr.CheckboxGroup(label="Selected scripts", choices=[])
                
                start_batch_btn = gr.Button("🚀 Start Batch Processing", variant="primary")

            with gr.Column():
                batch_log = gr.Textbox(label="Batch Status Log", lines=15, interactive=False)

    # ---------------- Toggle Logic ---------------- #
    def toggle_single_txt(use_txt):
        return gr.update(visible=not use_txt), gr.update(visible=use_txt)

    txt_input_toggle.change(toggle_single_txt, inputs=txt_input_toggle, outputs=[text_input, text_file_input])

    # ---------------- Click Events ---------------- #
    load_def_btn.click(load_defaults, outputs=[batch_ref_audio, batch_ref_text])
    scan_btn.click(scan_inputs, outputs=file_queue)
    
    def process_single(text, text_file, ref_audio, ref_text):
        final_text = text
        fname = None
        if text_file:
            fname = text_file
            with open(text_file, "r", encoding="utf-8") as f:
                final_text = f.read()
        return synthesize_speech(final_text, ref_audio, ref_text, input_file_name=fname)

    submit_btn.click(process_single, 
                    inputs=[text_input, text_file_input, ref_audio_input, ref_text_input], 
                    outputs=[output_audio, time_taken])

    start_batch_btn.click(run_batch_process, 
                         inputs=[file_queue, batch_ref_audio, batch_ref_text], 
                         outputs=batch_log)

# ------------------------------------------------ #
#   Launch Interface                               #
# ------------------------------------------------ #
iface.launch(
    theme=gr.themes.Soft(
        primary_hue="cyan",
        secondary_hue="blue",
        neutral_hue="gray",
        font=["Inter", "sans-serif"]
    ).set(
        body_background_fill="linear-gradient(135deg, #0f172a, #1e293b)",
        block_background_fill="#111827",
        block_title_text_color="white",
        block_border_color="#334155",
        body_text_color="#d1d5db",
        button_primary_background_fill="#0ea5e9",
        button_primary_text_color="white",
        button_secondary_background_fill="#1e293b",
        button_secondary_text_color="#e5e7eb"
    )
)
