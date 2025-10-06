import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import speech_recognition as sr
from transformers import pipeline
import time

# --- Page Configuration ---
st.set_page_config(page_title="Conversational Vibe Analyst", layout="wide")

# --- Model Loading Function ---
# Yeh function model ko tabhi load karega jab iski zaroorat padegi (memory bachane ke liye)
@st.cache_resource
def load_sentiment_model():
    """Loads the sentiment analysis model."""
    st.toast("Loading AI model for text analysis...")
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# --- App State Initialization ---
# Session state ko initialize karna zaroori hai
if 'user_text' not in st.session_state:
    st.session_state.user_text = ""
if 'current_emotion' not in st.session_state:
    st.session_state.current_emotion = "neutral"
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# --- Real-time Video Analysis Class ---
class EmotionVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        try:
            # Chehre par emotion detect karna
            analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, silent=True)
            if isinstance(analysis, list) and analysis:
                # Emotion ko session state mein store karna
                st.session_state.current_emotion = analysis[0]['dominant_emotion']
        except Exception as e:
            # Agar koi chehra na mile, toh error na dikhaye
            pass 
        return img

# --- UI Layout ---
st.title("Conversational Vibe Analyst 🎙️")
st.markdown("Apna camera aur microphone on karein, phir **'Press and Speak'** button dabakar baat karein.")

col1, col2 = st.columns([1.5, 2])

# Column 1: Live Video Feed
with col1:
    st.header("Live Feed")
    # Deployment ke liye RTC Configuration zaroori hai
    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    
    webrtc_streamer(
        key="aura-video-conversational",
        video_processor_factory=EmotionVideoTransformer,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# Column 2: Conversation and Analysis
with col2:
    st.header("Conversation")
    speak_button = st.button("🎤 Press and Speak", use_container_width=True)
    
    if speak_button:
        recognizer = sr.Recognizer()
        try:
            # Microphone se audio sunna
            with sr.Microphone() as source:
                st.info("Listening... Please speak now!")
                # Ambient noise ke liye adjust karna
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

            with st.spinner("Analyzing your complete vibe..."):
                # 1. Speech-to-Text Conversion
                user_text = recognizer.recognize_google(audio)
                st.session_state.user_text = user_text
                
                # 2. Text Sentiment Analysis (MODEL YAHAN LOAD HOGA)
                sentiment_analyzer = load_sentiment_model()
                text_sentiment = sentiment_analyzer(user_text)[0]
                
                # 3. Facial Emotion (Session state se lena)
                face_emotion = st.session_state.get('current_emotion', 'neutral')
                
                # Results ko display ke liye store karna
                st.session_state.analysis_results = {
                    "Face": face_emotion.capitalize(),
                    "Text Sentiment": f"{text_sentiment['label'].capitalize()} ({text_sentiment['score']:.2f})"
                }

        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio. Please try again.")
        except sr.WaitTimeoutError:
            st.warning("No speech detected. Please press the button and speak.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    # Results ko display karna
    if st.session_state.user_text:
        st.text_area("You said:", value=st.session_state.user_text, height=100, disabled=True)
    
    if st.session_state.analysis_results:
        st.markdown("---")
        st.subheader("Deep Analysis Breakdown")
        results = st.session_state.analysis_results
        st.markdown(f"😀 **Facial Emotion:** {results['Face']}")
        st.markdown(f"📝 **Word Sentiment:** {results['Text Sentiment']}")
