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
import time  # ⬅ Added for timing
from transformers import AutoModel

warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

# Function to load reference audio from URL
def load_audio_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        audio_data, sample_rate = sf.read(io.BytesIO(response.content))
        return sample_rate, audio_data
    return None, None

@spaces.GPU
def synthesize_speech(text, ref_audio, ref_text):
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

    # Save ref audio
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

    # Normalize
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

'''# Example Data (Multiple Examples)
EXAMPLES = [
    {
        "audio_name": "PAN_F (Happy)",
        "audio_url": "https://github.com/AI4Bharat/IndicF5/raw/refs/heads/main/prompts/PAN_F_HAPPY_00002.wav",
        "ref_text": "ਇੱਕ ਗ੍ਰਾਹਕ ਨੇ ਸਾਡੀ ਬੇਮਿਸਾਲ ਸੇਵਾ ਬਾਰੇ ਦਿਲੋਂਗਵਾਹੀ ਦਿੱਤੀ ਜਿਸ ਨਾਲ ਸਾਨੂੰ ਅਨੰਦ ਮਹਿਸੂਸ ਹੋਇਆ।",
        "synth_text": "मैं बिना किसी चिंता के अपने दोस्तों को अपने ऑटोमोबाइल एक्सपर्ट के पास भेज देता हूँ क्योंकि मैं जानता हूँ कि वह निश्चित रूप से उनकी सभी जरूरतों पर खरा उतरेगा।"
    },
    {
        "audio_name": "TAM_F (Happy)",
        "audio_url": "https://github.com/AI4Bharat/IndicF5/raw/refs/heads/main/prompts/TAM_F_HAPPY_00001.wav",
        "ref_text": "நான் நெனச்ச மாதிரியே அமேசான்ல பெரிய தள்ளுபடி வந்திருக்கு. கம்மி காசுக்கே அந்தப் புது சேம்சங் மாடல வாங்கிடலாம்.",
        "synth_text": "ഭക്ഷണത്തിന് ശേഷം തൈര് സാദം കഴിച്ചാൽ ഒരു ഉഷാറാണ്!"
    },
    {
        "audio_name": "MAR_F (WIKI)",
        "audio_url": "https://github.com/AI4Bharat/IndicF5/raw/refs/heads/main/prompts/MAR_F_WIKI_00001.wav",
        "ref_text": "दिगंतराव्दारे अंतराळ कक्षेतला कचरा चिन्हित करण्यासाठी प्रयत्न केले जात आहे.",
        "synth_text": "प्रारंभिक अंकुर छेदक. मी सोलापूर जिल्ह्यातील माळशिरस तालुक्यातील शेतकरी गणपत पाटील बोलतोय. माझ्या ऊस पिकावर प्रारंभिक अंकुर छेदक कीड आढळत आहे. क्लोरँट्रानिलीप्रोल (कोराजेन) वापरणे योग्य आहे का? त्याचे प्रमाण किती असावे?"
    },
    {
        "audio_name": "MAR_M (WIKI)",
        "audio_url": "https://github.com/AI4Bharat/IndicF5/raw/refs/heads/main/prompts/MAR_M_WIKI_00001.wav",
        "ref_text": "या प्रथाला एकोणीसशे पंचातर ईसवी पासून भारतीय दंड संहिताची धारा चारशे अठ्ठावीस आणि चारशे एकोणतीसच्या अन्तर्गत निषेध केला.",
        "synth_text": "जीवाणू करपा. मी अहमदनगर जिल्ह्यातील राहुरी गावातून बाळासाहेब जाधव बोलतोय. माझ्या डाळिंब बागेत जीवाणू करपा मोठ्या प्रमाणात दिसतोय. स्ट्रेप्टोसायक्लिन आणि कॉपर ऑक्सिक्लोराईड फवारणीसाठी योग्य प्रमाण काय असावे?"
    },
    {
        "audio_name": "KAN_F (Happy)",
        "audio_url": "https://github.com/AI4Bharat/IndicF5/raw/refs/heads/main/prompts/KAN_F_HAPPY_00001.wav",
        "ref_text": "ನಮ್‌ ಫ್ರಿಜ್ಜಲ್ಲಿ  ಕೂಲಿಂಗ್‌ ಸಮಸ್ಯೆ ಆಗಿ ನಾನ್‌ ಭಾಳ ದಿನದಿಂದ ಒದ್ದಾಡ್ತಿದ್ದೆ, ಆದ್ರೆ ಅದ್ನೀಗ ಮೆಕಾನಿಕ್ ಆಗಿರೋ ನಿಮ್‌ ಸಹಾಯ್ದಿಂದ ಬಗೆಹರಿಸ್ಕೋಬೋದು ಅಂತಾಗಿ ನಿರಾಳ ಆಯ್ತು ನಂಗೆ.",
        "synth_text": "চেন্নাইয়ের শেয়ারের অটোর যাত্রীদের মধ্যে খাবার ভাগ করে খাওয়াটা আমার কাছে মন খুব ভালো করে দেওয়া একটা বিষয়।"
    },
]


# Preload all example audios
for example in EXAMPLES:
    sample_rate, audio_data = load_audio_from_url(example["audio_url"])
    example["sample_rate"] = sample_rate
    example["audio_data"] = audio_data'''


# Define Gradio interface with layout adjustments
with gr.Blocks() as iface:
    gr.Markdown(
        """
        # **Spider『X』(T2S) Indic_F5 Speech Model - 1.0** 
        
        Generate speech using a reference prompt audio and its corresponding text.
        """
    )
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Text to Synthesize", placeholder="Enter the text to convert to speech...", lines=3)
            ref_audio_input = gr.Audio(type="numpy", label="Reference Prompt Audio")
            ref_text_input = gr.Textbox(label="Text in Reference Prompt Audio", placeholder="Enter the transcript of the reference audio...", lines=2)
            submit_btn = gr.Button("🎤 Generate Speech", variant="primary")
        
        with gr.Column():
            output_audio = gr.Audio(label="Generated Speech", type="numpy")
            time_taken = gr.Label(label="Total Time Taken (HH:MM:SS)") 

    
    '''# Add multiple examples
    examples = [
        [ex["synth_text"], (ex["sample_rate"], ex["audio_data"]), ex["ref_text"]] for ex in EXAMPLES
    ]
    
    gr.Examples(
        examples=examples,
        inputs=[text_input, ref_audio_input, ref_text_input],
        label="Choose an example:"
    )'''
    theme=gr.themes.Soft()
    submit_btn.click(
    synthesize_speech,
    inputs=[text_input, ref_audio_input, ref_text_input],
    outputs=[output_audio, time_taken] 
)


iface.launch()
