import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import speech_recognition as sr
from gtts import gTTS
import base64
import librosa
from transformers import pipeline

st.set_page_config(page_title="Conversational AI", layout="wide")

# This function will now be called only when the button is pressed
@st.cache_resource
def load_sentiment_model():
    """Loads the sentiment analysis model."""
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Initialize session state variables
if 'user_text' not in st.session_state: st.session_state['user_text'] = ""
if 'current_emotion' not in st.session_state: st.session_state['current_emotion'] = "neutral"
if 'analysis_results' not in st.session_state: st.session_state['analysis_results'] = None

class EmotionVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        try:
            # Analyze emotions
            analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, silent=True)
            if isinstance(analysis, list) and analysis:
                st.session_state['current_emotion'] = analysis[0]['dominant_emotion']
        except Exception:
            pass 
        return img

st.title("Conversational Vibe Analyst 🎙️")

col1, col2 = st.columns([1.5, 2])

with col1:
    st.header("Live Feed")
    webrtc_streamer(
        key="aura-video-conv",
        video_processor_factory=EmotionVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

with col2:
    st.header("Conversation")
    speak_button = st.button("🎤 Press and Speak", use_container_width=True)
    
    if speak_button:
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                st.info("Listening... Speak now!")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

            with st.spinner("Analyzing your complete vibe..."):
                # 1. Speech-to-Text
                user_text = recognizer.recognize_google(audio)
                st.session_state['user_text'] = user_text
                
                # 2. Text Sentiment Analysis (MODEL IS LOADED HERE, NOT AT STARTUP)
                sentiment_analyzer = load_sentiment_model()
                text_sentiment = sentiment_analyzer(user_text)[0]
                
                # 3. Facial Emotion (grab from state)
                face_emotion = st.session_state.get('current_emotion', 'neutral')
                
                # Store results for display
                st.session_state['analysis_results'] = {
                    "Face": face_emotion.capitalize(),
                    "Text Sentiment": f"{text_sentiment['label']} ({text_sentiment['score']:.2f})"
                }

        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    if st.session_state.user_text:
        st.text_area("You said:", value=st.session_state.user_text, height=100, disabled=True)
    
    if st.session_state.analysis_results:
        st.markdown("---")
        st.subheader("Deep Analysis Breakdown")
        results = st.session_state.analysis_results
        st.markdown(f"😀 **Facial Emotion:** {results['Face']}")
        st.markdown(f"📝 **Word Sentiment:** {results['Text Sentiment']}")
