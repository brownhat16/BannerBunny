import streamlit as st
import requests
import base64
import time

# API Configuration
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Banner Generation UI", layout="wide")
st.title("🎨 Banner Generation UI")

# Sidebar Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Generate Banner", "Check Status"])

if page == "Generate Banner":
    st.subheader("📝 Generate Banner from Prompt")
    prompt = st.text_area("Enter your banner description:", height=150, placeholder="e.g., 'A vibrant red banner for electronics sale'")
    width = st.slider("Image Width", min_value=320, max_value=2048, value=1024, step=32)
    height = st.slider("Image Height", min_value=240, max_value=1440, value=768, step=32)
    submit_button = st.button("Generate Banner")

    if submit_button and prompt:
        with st.spinner("Generating banner..."):
            payload = {
                "prompt": prompt,
                "width": width,
                "height": height
            }
            try:
                response = requests.post(f"{API_URL}/generate-banner", json=payload)
                result = response.json()

                if result["status"] == "success":
                    st.success("✅ Banner generation successful!")
                    st.markdown("### 🧾 Generated Metadata")
                    st.json(result["structured_data"])
                    st.markdown("### 🎨 FLUX Prompt")
                    st.code(result["flux_prompt"], language="text")
                    if result.get("image_base64"):
                        st.markdown("### 🖼️ Generated Banner")
                        image_data = base64.b64decode(result["image_base64"])
                        st.image(image_data, caption="Generated Banner", use_column_width=True)
                elif result["status"] == "partial_success":
                    st.warning("⚠️ Metadata extracted but image generation failed.")
                    st.json(result["structured_data"])
                else:
                    st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"🚨 Failed to connect to backend: {str(e)}")

elif page == "Check Status":
    st.subheader("🔍 Check Async Banner Job Status")
    request_id = st.text_input("Enter Request ID", placeholder="async_req_123456789")
    check_button = st.button("Check Status")

    if check_button and request_id:
        with st.spinner("Fetching job status..."):
            try:
                response = requests.get(f"{API_URL}/status/{request_id}")
                if response.status_code == 200:
                    status = response.json()
                    st.json(status)
                    if status["status"] == "completed" and status["result"].get("image_base64"):
                        image_data = base64.b64decode(status["result"]["image_base64"])
                        st.image(image_data, caption="Generated Banner", use_column_width=True)
                else:
                    st.error("🚫 Job not found or server error.")
            except Exception as e:
                st.error(f"🚨 Connection error: {str(e)}")
