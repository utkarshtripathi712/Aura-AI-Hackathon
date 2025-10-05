import streamlit as st
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import speech_recognition as sr
from gtts import gTTS
import base64
import librosa
from transformers import pipeline
import av
import threading
import time

# --- UNIVERSAL HELPER FUNCTIONS ---
def map_emotion_to_emoji(emotion):
    emotion_emoji_map = {'happy': '😄', 'sad': '😢', 'angry': '😠', 'neutral': '😐', 'surprise': '😮', 'fear': '😨', 'disgust': '🤢'}
    return emotion_emoji_map.get(emotion, '🤔')

# --- HOME PAGE RENDER FUNCTION ---
def render_home_page():
    st.title("✨ Aura AI")
    st.header("Understand Before You Ask")
    st.markdown("""
    Kya aapne kabhi socha hai ki jab hum kisi se pehli baar milte hain - ek job interview mein, ek client meeting mein, ya ek date par - toh kitna kuch unkaha reh jaata hai? Aura AI aapko saamne waale insaan ki personality aur emotional state ki ek jhalak deta hai, bina ek bhi sawaal puche.
    """)
    st.markdown("---")
    st.header("Revolutionary Capabilities")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("👁️ Instant Personality Reading")
        st.write("Analyze personality traits, confidence levels, and communication styles in real-time.")
    with col2:
        st.subheader("❤️ Emotional Intelligence")
        st.write("Detect stress indicators, emotional states, and behavioral patterns without direct questions.")
    with col3:
        st.subheader("🔄 Multi-Context Analysis")
        st.write("Specialized insights for interviews, meetings, dating, and general conversations.")
    st.markdown("---")
    st.info("👈 Sidebar se koi bhi live demo select karein aur Aura AI ka power experience karein!")

# --- STATIC VIBE CHECK PAGE RENDER FUNCTION ---
def render_static_vibe_check():
    st.title("Aura AI: Static Vibe Check 📸")
    st.markdown("##### Take a snapshot and get a detailed breakdown of your emotional vibe.")
    st.markdown("---")
    
    def generate_insight(emotions):
        dominant_emotion = max(emotions, key=emotions.get)
        confidence = emotions.get('happy', 0) + emotions.get('neutral', 0)
        stress = emotions.get('angry', 0) + emotions.get('fear', 0)
        if dominant_emotion == 'happy' and confidence > 60: return "You're radiating positivity and confidence! Great energy. 👍"
        if dominant_emotion == 'neutral' and confidence > 70: return "You appear calm, composed, and focused. Very professional."
        if dominant_emotion == 'sad': return "You seem a bit down. Remember to take a moment for yourself."
        if stress > 30: return "High-stress levels detected. A deep breath could make a difference."
        if dominant_emotion == 'surprise': return "Engaged and curious! Your expression shows you're locked in."
        return "Your expression is a unique mix! Aura is learning more about you."

    col1, col2 = st.columns([1, 1.2], gap="large")
    with col1:
        st.header("Step 1: Capture Your Vibe")
        img_file_buffer = st.camera_input("Take a photo", key="camera_input", help="Ensure your face is well-lit.")
        if img_file_buffer:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            st.session_state['captured_image'] = cv2_img
            st.image(cv2_img, channels="BGR", caption="Your snapshot!")
    with col2:
        st.header("Step 2: Get Your Analysis")
        analyze_button = st.button("Read My Vibe", type="primary", use_container_width=True, disabled=not('captured_image' in st.session_state))
        if analyze_button and 'captured_image' in st.session_state:
            with st.spinner("Reading your vibe... ✨"):
                try:
                    analysis = DeepFace.analyze(img_path=st.session_state['captured_image'], actions=['emotion'], enforce_detection=False)
                    if analysis:
                        result = analysis[0]
                        dominant_emotion = result['dominant_emotion']
                        emotions = result['emotion']
                        st.success("**Analysis Complete!**")
                        emoji = map_emotion_to_emoji(dominant_emotion)
                        st.metric(label="Dominant Vibe", value=f"{dominant_emotion.capitalize()} {emoji}")
                        insight = generate_insight(emotions)
                        st.markdown(f"> *{insight}*")
                        st.markdown("---")
                        st.subheader("Emotional Breakdown")
                        df_emotions = pd.DataFrame(emotions.items(), columns=['Emotion', 'Percentage'])
                        st.bar_chart(df_emotions.set_index('Emotion'))
                    else:
                        st.error("Could not find a face in the image. Please try again.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        elif analyze_button:
            st.warning("Please take a photo first!", icon="⚠️")

# --- CONVERSATIONAL ANALYST PAGE RENDER FUNCTION ---
def render_conversational_analyst():
    st.title("Aura AI: Conversational Analyst 🎙️")
    st.markdown("I can see your expressions, hear your tone, and understand your words. **Press the button and speak.**")
    
    @st.cache_resource
    def load_sentiment_model():
        return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    sentiment_analyzer = load_sentiment_model()

    if 'user_text' not in st.session_state: st.session_state['user_text'] = ""
    if 'ai_response' not in st.session_state: st.session_state['ai_response'] = ""
    if 'current_emotion' not in st.session_state: st.session_state['current_emotion'] = "neutral"
    if 'analysis_results' not in st.session_state: st.session_state['analysis_results'] = None

    class EmotionVideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            try:
                analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, silent=True)
                if isinstance(analysis, list) and analysis:
                    st.session_state['current_emotion'] = analysis[0]['dominant_emotion']
                    region = analysis[0]['region']
                    x, y, w, h = region['x'], region['y'], region['w'], region['h']
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, st.session_state.current_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception: pass
            return img

    def analyze_voice_tone(audio_file):
        try:
            y, sr_audio = librosa.load(audio_file)
            pitch = np.mean(librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')))
            energy = np.mean(librosa.feature.rms(y=y))
            tone = "Neutral"
            if pitch > 160: tone = "High-Pitched"
            elif pitch < 100 and pitch > 1: tone = "Low-Pitched"
            if energy > 0.05: tone += " / High Energy"
            else: tone += " / Low Energy"
            return tone
        except Exception: return "Undetermined"

    def generate_fused_insight(face_emotion, voice_tone, text_sentiment):
        face = face_emotion.lower(); sentiment = text_sentiment['label'].lower()
        insight = f"Facial expression is **{face}**. Word sentiment is **{sentiment}**. Voice tone is **{voice_tone}**."
        if (face in ["happy", "neutral"]) and sentiment == "negative": insight += "\n\n**Insight:** Mismatch detected. While you appear calm, your words express negativity."
        elif face == "sad" and sentiment == "positive": insight += "\n\n**Insight:** You're saying positive things, but your expression suggests otherwise."
        elif "high energy" in voice_tone.lower() and (face == "angry" or sentiment == "negative"): insight += "\n\n**Insight:** High energy in your voice combined with your words/expression suggests strong feelings."
        else: insight += "\n\n**Insight:** All signs are aligned, indicating genuineness."
        return insight

    def text_to_speech_autoplay(text):
        tts = gTTS(text=text, lang='en'); tts.save("response.mp3")
        with open("response.mp3", "rb") as f: data = f.read()
        b64 = base64.b64encode(data).decode()
        return f'<audio controls autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'

    col1, col2 = st.columns([1.5, 2])
    with col1:
        st.header("Live Feed")
        RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        webrtc_streamer(key="aura-video", video_processor_factory=EmotionVideoTransformer, rtc_configuration=RTC_CONFIGURATION, media_stream_constraints={"video": True, "audio": False}, async_processing=True)
    with col2:
        st.header("Conversation")
        if st.button("🎤 Press and Speak", use_container_width=True):
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                st.info("Listening... Speak now!"); recognizer.adjust_for_ambient_noise(source)
                try:
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    with st.spinner("Analyzing your complete vibe..."):
                        with open("user_speech.wav", "wb") as f: f.write(audio.get_wav_data())
                        user_text = recognizer.recognize_google(audio)
                        st.session_state['user_text'] = user_text
                        voice_tone = analyze_voice_tone("user_speech.wav")
                        text_sentiment = sentiment_analyzer(user_text)[0]
                        face_emotion = st.session_state.get('current_emotion', 'neutral')
                        fused_insight = generate_fused_insight(face_emotion, voice_tone, text_sentiment)
                        st.session_state['ai_response'] = "Thanks for sharing. " + fused_insight.split("\n\n**Insight:**")[0]
                        st.session_state['analysis_results'] = {"Face": f"{face_emotion.capitalize()}", "Voice Tone": voice_tone, "Text Sentiment": f"{text_sentiment['label']} ({text_sentiment['score']:.2f})", "Fused Insight": fused_insight}
                except Exception as e: st.error(f"An error occurred: {e}")
        if st.session_state.user_text: st.text_area("You said:", value=st.session_state.user_text, height=100, disabled=True)
        if st.session_state.ai_response: st.markdown(text_to_speech_autoplay(st.session_state.ai_response), unsafe_allow_html=True)
        if st.session_state.analysis_results:
            st.markdown("---"); st.subheader("Deep Analysis Breakdown")
            results = st.session_state.analysis_results
            st.markdown(f"😀 **Facial Emotion:** {results['Face']}"); st.markdown(f"🔊 **Voice Tone:** {results['Voice Tone']}"); st.markdown(f"📝 **Word Sentiment:** {results['Text Sentiment']}")
            st.info(f"**🧠 Synthesized Insight:**\n\n{results['Fused Insight'].split('**Insight:**')[-1].strip()}")

# --- LIVE DASHBOARD PAGE RENDER FUNCTION ---
def render_live_dashboard():
    st.title("Aura AI: Live Dashboard ✨")
    st.markdown("##### Turn on your camera and see your emotional vibe analyzed in real-time.")
    
    results_lock = threading.Lock()
    latest_results = {"dominant_emotion": "neutral", "emotions": {'neutral': 100.0}, "insight": "Waiting for analysis...", "confidence": 0, "calmness": 0, "stress": 0}

    def generate_insight(emotions):
        dominant_emotion = max(emotions, key=emotions.get)
        confidence = emotions.get('happy', 0) + emotions.get('neutral', 0)
        stress = emotions.get('angry', 0) + emotions.get('fear', 0)
        if dominant_emotion == 'happy' and confidence > 60: return "You're radiating positivity and confidence! Great energy. 👍"
        if dominant_emotion == 'neutral' and confidence > 70: return "You appear calm, composed, and focused. Very professional."
        if dominant_emotion == 'sad': return "You seem a bit down. Remember to take a moment for yourself."
        if stress > 30: return "High-stress levels detected. A deep breath could make a difference."
        if dominant_emotion == 'surprise': return "Engaged and curious! Your expression shows you're locked in."
        return "Your expression is a unique mix! Aura is learning more about you."

    class EmotionTransformer(VideoTransformerBase):
        def __init__(self):
            self.frame_counter = 0; self.analysis_interval = 10; self.last_known_face = None
        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            img = frame.to_ndarray(format="bgr24")
            if self.frame_counter % self.analysis_interval == 0:
                try:
                    analysis = DeepFace.analyze(img_path=img, actions=['emotion'], enforce_detection=False, silent=True)
                    if isinstance(analysis, list) and len(analysis) > 0:
                        result = analysis[0]; self.last_known_face = result
                        with results_lock:
                            latest_results["dominant_emotion"] = result['dominant_emotion']
                            latest_results["emotions"] = result['emotion']
                            latest_results["insight"] = generate_insight(result['emotion'])
                            emotions = result['emotion']
                            latest_results["confidence"] = (emotions.get('happy', 0) + emotions.get('neutral', 0))
                            stress_val = (emotions.get('angry', 0) + emotions.get('fear', 0) + emotions.get('disgust', 0))
                            latest_results["stress"] = stress_val
                            latest_results["calmness"] = 100 - stress_val
                except Exception: pass
            if self.last_known_face:
                try:
                    region = self.last_known_face['region']; x, y, w, h = region['x'], region['y'], region['w'], region['h']
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    emotion_text = f"{self.last_known_face['dominant_emotion'].capitalize()} {map_emotion_to_emoji(self.last_known_face['dominant_emotion'])}"
                    (text_w, text_h), _ = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(img, (x, y - text_h - 10), (x + text_w, y - 5), (0, 255, 0), -1)
                    cv2.putText(img, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                except: pass
            self.frame_counter += 1
            return img

    st.markdown("---")
    col1, col2 = st.columns([2, 1.2])
    with col1:
        st.header("Live Camera Feed")
        RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        webrtc_ctx = webrtc_streamer(key="aura-live", video_processor_factory=EmotionTransformer, rtc_configuration=RTC_CONFIGURATION, media_stream_constraints={"video": True, "audio": False}, async_processing=True)
        if not webrtc_ctx.state.playing: st.info("Click 'START' to turn on your camera.", icon="▶️")
    with col2:
        st.header("Real-time Dashboard")
        if not webrtc_ctx.state.playing: st.warning("Dashboard is idle. Start the camera feed.", icon="📊")
        insight_placeholder, metric_placeholder, progress_placeholder, chart_placeholder = st.empty(), st.empty(), st.empty(), st.empty()
        while webrtc_ctx.state.playing:
            with results_lock: results = latest_results.copy()
            with insight_placeholder.container(): st.subheader("Your Vibe Insight"); st.markdown(f"> *{results['insight']}*")
            with metric_placeholder.container(): st.subheader("Dominant Vibe"); emoji = map_emotion_to_emoji(results['dominant_emotion']); st.metric(label="Current Emotion", value=f"{results['dominant_emotion'].capitalize()} {emoji}")
            with progress_placeholder.container(): st.subheader("Key Metrics"); st.progress(int(results['confidence']), text=f"Confidence: {int(results['confidence'])}%"); st.progress(int(results['calmness']), text=f"Calmness: {int(results['calmness'])}%"); st.progress(int(results['stress']), text=f"Stress: {int(results['stress'])}%")
            with chart_placeholder.container(): st.subheader("Full Emotional Spectrum"); df_emotions = pd.DataFrame(results['emotions'].items(), columns=['Emotion', 'Percentage']); st.bar_chart(df_emotions.set_index('Emotion'))
            time.sleep(0.2)


# --- MAIN APP LOGIC ---
st.set_page_config(page_title="Aura AI", page_icon="🧠", layout="wide")

if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Sidebar for navigation
with st.sidebar:
    st.title("🚀 Aura AI Demos")
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = "Home"
    if st.button("📸 Static Vibe Check", use_container_width=True):
        st.session_state.page = "Static Vibe Check"
    if st.button("🎙️ Conversational Analyst", use_container_width=True):
        st.session_state.page = "Conversational Analyst"
    if st.button("✨ Live Dashboard", use_container_width=True):
        st.session_state.page = "Live Dashboard"

# Render the selected page
if st.session_state.page == "Home":
    render_home_page()
elif st.session_state.page == "Static Vibe Check":
    render_static_vibe_check()
elif st.session_state.page == "Conversational Analyst":
    render_conversational_analyst()
elif st.session_state.page == "Live Dashboard":
    render_live_dashboard()