import streamlit as st
from audio_recorder_streamlit import audio_recorder
import google.generativeai as genai
from gtts import gTTS
import io

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
    st.set_page_config(page_title="Gari AI Voice Agent", page_icon="🎙️")
    
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
                    
                    st.write("**Transcribed Text:**", transcribed_text)
                    st.write("**AI Response:**", ai_response)

                    # Fast audio playback
                    audio_fp = text_to_speech_bytes(ai_response)
                    st.audio(audio_fp, format="audio/mp3")

                except Exception as err:
                    st.error(f"Error: {err}")
    else:
        st.info("👈 Please enter your Gemini API key in the sidebar to get started.")

if __name__ == "__main__":
    main()