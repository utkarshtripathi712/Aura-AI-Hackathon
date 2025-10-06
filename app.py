import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="Aura AI - Home",
    page_icon="🧠",
    layout="wide"
)

# --- Hero Section ---
st.title("✨ Aura AI")
st.header("Understand Before You Ask")

st.markdown("""
Kya aapne kabhi socha hai ki jab hum kisi se pehli baar milte hain - ek job interview mein, ek client meeting mein, ya ek date par - toh kitna kuch unkaha reh jaata hai? Aura AI aapko saamne waale insaan ki personality aur emotional state ki ek jhalak deta hai, bina ek bhi sawaal puche.
""")
st.markdown("---")


# --- Revolutionary Capabilities Section ---
st.header("Revolutionary Capabilities")
st.markdown("Advanced AI that understands personality, emotions, and behavioral patterns through cutting-edge analysis.")

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

# --- Call to Action / How to Use ---
st.header("Try the Live Demos")
st.info(
    """
    **👈 Sidebar se koi bhi demo select karein aur Aura AI ka power experience karein!**

    - **Conversational Vibe Analyst:** Microphone aur Camera ka istemaal karke complete analysis.
    - **Live Vibe Analysis:** Sirf Camera se real-time emotional dashboard.
    """
)
