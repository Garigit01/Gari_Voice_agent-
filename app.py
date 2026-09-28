import streamlit as st
from audio_recorder_streamlit import audio_recorder
import google.generativeai as genai
from gtts import gTTS
import io

# Page Configuration
st.set_page_config(
    page_title="Gari AI Voice Agent",
    page_icon="🎙️"
)

# Custom Styling (Gradient hataya gaya, clean solid cards add kiye gaye)
st.markdown("""
<style>
    /* Clean dark background (No gradient) */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }

    /* Input Card - Slate/Blue Theme */
    .input-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-left: 5px solid #38bdf8;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 14px;
        margin-bottom: 12px;
    }
    .input-label {
        color: #38bdf8;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .input-text {
        color: #f1f5f9;
        font-size: 1.05rem;
        line-height: 1.5;
    }

    /* Output Card - Forest/Emerald Theme */
    .output-card {
        background-color: #064e3b;
        border: 1px solid #047857;
        border-left: 5px solid #34d399;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 12px;
        margin-bottom: 14px;
    }
    .output-label {
        color: #34d399;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .output-text {
        color: #ecfdf5;
        font-size: 1.05rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# 1. Latest Gemini 3.8 Flash model configure karein
@st.cache_resource
def get_gemini_model(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-3.8-flash")

# 2. Audio process function (Gemini direct audio bytes process karega)
def process_audio_with_gemini(model, audio_bytes):
    audio_part = {
        "mime_type": "audio/wav",
        "data": audio_bytes
    }
    
    prompt = """
    Listen to the audio carefully.
    1. Transcribe the user's speech.
    2. Provide a short, direct and friendly conversational response (max 2 sentences).
    
    Format strictly as:
    TRANSCRIPT: <transcribed text>
    RESPONSE: <your AI response>
    """
    
    response = model.generate_content([audio_part, prompt])
    raw_text = response.text
    
    if "TRANSCRIPT:" in raw_text and "RESPONSE:" in raw_text:
        parts = raw_text.split("RESPONSE:")
        transcribed_text = parts[0].replace("TRANSCRIPT:", "").strip()
        ai_response = parts[1].strip()
    else:
        transcribed_text = "Audio processed"
        ai_response = raw_text.strip()
        
    return transcribed_text, ai_response

# 3. Fast In-memory TTS (no disk read/write delay)
def text_to_speech_bytes(text):
    fp = io.BytesIO()
    tts = gTTS(text=text, lang="en", slow=False)
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

def main():
    st.sidebar.title("API KEY CONFIGURATION")
    api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")

    st.title("🎙️ Gari AI Voice Agent")
    st.caption("✨ Speak naturally. Let Gari handle the rest.")

    if api_key:
        try:
            model = get_gemini_model(api_key)
        except Exception as e:
            st.error(f"Error configuring API Key: {e}")
            return

        st.write("Click the mic below to record your voice:")
        recorded_audio = audio_recorder()

        if recorded_audio:
            with st.spinner("⚡ Thinking & responding..."):
                try:
                    transcribed_text, ai_response = process_audio_with_gemini(model, recorded_audio)
                    
                    # 1. Input Text Card (Blue Accent)
                    st.markdown(f"""
                        <div class="input-card">
                            <div class="input-label">👤 Transcribed Text</div>
                            <div class="input-text">{transcribed_text}</div>
                        </div>
                    """, unsafe_allow_html=True)

                    # 2. Output Text Card (Green/Emerald Accent)
                    st.markdown(f"""
                        <div class="output-card">
                            <div class="output-label">🤖 AI Response</div>
                            <div class="output-text">{ai_response}</div>
                        </div>
                    """, unsafe_allow_html=True)

                    # 3. Fast audio playback with AUTO-PLAY
                    audio_fp = text_to_speech_bytes(ai_response)
                    st.audio(audio_fp, format="audio/mp3", autoplay=True)

                except Exception as err:
                    st.error(f"Error: {err}")
    else:
        st.info("👈 Please enter your Gemini API key in the sidebar to get started.")

if __name__ == "__main__":
    main()