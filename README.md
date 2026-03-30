import streamlit as st
import openai  # For Grok API (compatible)
from wolframalpha import Client as WolframClient
from gtts import gTTS
import pygame
import io
import base64
from PIL import Image
import cv2
import numpy as np
import mathjax_renderer  # For LaTeX

# Page config
st.set_page_config(page_title="AI Teacher App", layout="wide")
st.title("🤖 AI Teacher: Step-by-Step Learning Assistant")

# Sidebar for APIs (use secrets in production)
st.sidebar.header("API Keys")
grok_api_key = st.sidebar.text_input("xAI Grok API Key", type="password")
wolfram_appid = st.sidebar.text_input("Wolfram Alpha AppID", type="password")

# Initialize clients
@st.cache_resource
def init_clients():
    if grok_api_key:
        openai.api_key = grok_api_key
        openai.api_base = "https://api.x.ai/v1"  # Grok endpoint
    if wolfram_appid:
        return WolframClient(wolfram_appid)
    return None

wolfram_client = init_clients()

# Language options (Indian focus)
languages = ["en", "hi", "ta", "te"]  # English, Hindi, Tamil, Telugu
lang = st.sidebar.selectbox("Audio Language", languages)

# Audio player helper
def play_audio(audio_bytes, format="wav"):
    b64 = base64.b64encode(audio_bytes).decode()
    st.audio(f"data:audio/{format};base64,{b64}", autoplay=False)

# TTS function
def generate_tts(text):
    tts = gTTS(text=text, lang=lang, slow=False)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp.read()

# File analysis helper (image/video)
def analyze_media(file):
    if file.type.startswith("image/"):
        img = Image.open(file)
        st.image(img, caption="Uploaded Image")
        # Base64 for Grok vision
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        return [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}]
    elif file.type.startswith("video/"):
        # Save temp video
        tfile = io.BytesIO(file.read())
        with open("temp.mp4", "wb") as f:
            f.write(tfile.getvalue())
        cap = cv2.VideoCapture("temp.mp4")
        frames = []
        while len(frames) < 5:  # Sample frames
            ret, frame = cap.read()
            if not ret: break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        st.image(frames[0], caption="Video Frame Sample")
        return "Analyzed video frames; describe content below."  # Simplified; enhance with full vision
    else:
        return f"File content: {file.read().decode()}"

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📚 Explain Topic/Math", "🖼️ Analyze Media", "🔨 Build Project", "🎯 Custom Prompt"])

with tab1:
    input_text = st.text_area("Enter topic/math problem (e.g., 'Solve x^2 + 2x + 1 = 0 step by step')")
    if st.button("Explain Step-by-Step", key="explain"):
        if wolfram_appid and "solve" in input_text.lower():
            try:
                res = wolfram_client.query(input_text)
                steps = next((p for p in res.pods if "Step-by-Step" in p["@title"]), None)
                if steps:
                    st.write("**Math Steps:**")
                    for sub in steps.subpods:
                        st.image(sub.img.src)
                        st.latex(mathjax_renderer.render(sub.img.alt))  # LaTeX
                else:
                    st.write("Wolfram Result:", next(res.results).text)
            except:
                st.error("Wolfram error; using Grok.")
        # Grok for general explanation
        if grok_api_key:
            response = openai.ChatCompletion.create(
                model="grok-beta",  # Or latest
                messages=[{"role": "system", "content": "Explain like a patient teacher: step-by-step, structured, educational."},
                          {"role": "user", "content": input_text}]
            )
            explanation = response.choices[0].message.content
            st.markdown(explanation)
            audio = generate_tts(explanation[:500])  # Truncate for TTS
            if st.button("🔊 Play Explanation Audio"):
                play_audio(audio)

with tab2:
    uploaded_file = st.file_uploader("Upload photo/file/video", type=["png", "jpg", "mp4", "txt", "pdf"])
    if uploaded_file and st.button("Analyze & Explain"):
        media_content = analyze_media(uploaded_file)
        if grok_api_key:
            msg = [{"role": "user", "content": [
                {"type": "text", "text": "Analyze and explain this educationally: describe, teach concepts from it."},
                *([media_content] if isinstance(media_content, list) else [{"type": "text", "text": str(media_content)}])
            ]}]
            response = openai.ChatCompletion.create(model="grok-vision-beta", messages=msg)  # Vision model
            st.markdown(response.choices[0].message.content)
            audio = generate_tts(response.choices[0].message.content[:500])
            if st.button("🔊 Play Analysis Audio"):
                play_audio(audio)

with tab3:
    project_prompt = st.text_area("Describe project (e.g., 'Python calculator with GUI')")
    if st.button("Generate Full Project Code"):
        if grok_api_key:
            response = openai.ChatCompletion.create(
                model="grok-beta",
                messages=[{"role": "system", "content": "Build a complete, working project: full code, explanations, run instructions."},
                          {"role": "user", "content": project_prompt}]
            )
            code = response.choices[0].message.content
            st.code(code, language="python")
            audio = generate_tts(f"Project generated: {project_prompt}")
            if st.button("🔊 Hear Project Summary"):
                play_audio(audio)

with tab4:
    custom_prompt = st.text_area("Any custom request")
    if st.button("Process with AI Teacher"):
        if grok_api_key:
            response = openai.ChatCompletion.create(
                model="grok-beta",
                messages=[{"role": "system", "content": "Act as a teacher: explain, solve, analyze step-by-step."},
                          {"role": "user", "content": custom_prompt}]
            )
            st.markdown(response.choices[0].message.content)
            audio = generate_tts(response.choices[0].message.content[:500])
            if st.button("🔊 Play Response"):
                play_audio(audio)
