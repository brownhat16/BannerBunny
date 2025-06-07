import streamlit as st
import requests
import base64
import time
from io import BytesIO
from PIL import Image

# Configuration
API_BASE_URL = "https://bannerbunny.onrender.com"
HEADERS = {"Content-Type": "application/json"}

def generate_banner(payload, async_mode=False):
    endpoint = "/generate-banner-async" if async_mode else "/generate-banner"
    try:
        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            json=payload,
            headers=HEADERS
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def poll_async_status(request_id):
    try:
        response = requests.get(f"{API_BASE_URL}/status/{request_id}")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def main():
    st.set_page_config(page_title="BannerBunny Generator", layout="wide")
    
    st.title("🚀 BannerBunny Creative Studio")
    st.markdown("### AI-Powered Advertising Banner Generation")
    
    with st.sidebar:
        st.header("⚙️ Generation Settings")
        prompt = st.text_area("Creative Brief", help="Describe your banner concept in detail")
        width = st.slider("Width", 512, 2048, 1024)
        height = st.slider("Height", 512, 2048, 768)
        seed = st.number_input("Seed (for reproducibility)", min_value=0, value=None)
        async_mode = st.toggle("Async Processing", value=False)
        advanced = st.checkbox("Show Advanced Options")
        
        if advanced:
            st.subheader("Advanced Parameters")
            num_inference_steps = st.slider("Inference Steps", 10, 50, 28)
            guidance_scale = st.slider("Guidance Scale", 1.0, 5.0, 3.5)
        else:
            num_inference_steps = 28
            guidance_scale = 3.5
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("✨ Generate Banner", type="primary"):
            if not prompt:
                st.error("Please enter a creative brief")
                return
            
            payload = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed or None
            }
            
            with st.spinner("🚧 Generating your masterpiece..."):
                result = generate_banner(payload, async_mode)
                
                if "error" in result:
                    st.error(f"API Error: {result['error']}")
                    return
                
                if async_mode:
                    request_id = result.get("request_id")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    while True:
                        status = poll_async_status(request_id)
                        if status.get("status") == "completed":
                            result = status.get("result")
                            break
                        if status.get("status") == "failed":
                            st.error("Generation failed")
                            return
                        
                        progress = float(status.get("progress", "0").strip('%'))/100
                        progress_bar.progress(progress, text=status.get("progress"))
                        time.sleep(1)
                
                if result.get("image_base64"):
                    image_data = base64.b64decode(result["image_base64"])
                    image = Image.open(BytesIO(image_data))
                    
                    st.image(image, caption="Generated Banner", use_column_width=True)
                    
                    # Download Section
                    buf = BytesIO()
                    image.save(buf, format="PNG")
                    st.download_button(
                        label="📥 Download Banner",
                        data=buf.getvalue(),
                        file_name="bannerbunny_output.png",
                        mime="image/png",
                        type="primary"
                    )
                    
                    # Metadata
                    with st.expander("🧠 Generation Details"):
                        st.json({
                            "Processing Time": f"{result.get('processing_time', 0):.2f}s",
                            "Request ID": result.get("request_id"),
                            "Flux Prompt": result.get("flux_prompt")
                        })
                        
                        st.subheader("Structured Metadata")
                        st.write(result.get("structured_data", {}))
                else:
                    st.error("Generation failed - no image returned")
    
    with col2:
        st.markdown("### 📚 Quick Tips")
        st.markdown("""
        - Use descriptive prompts for best results
        - Start with 1024x768 resolution
        - Use async mode for complex generations
        - Experiment with seeds for variations
        """)
        
        st.markdown("### 🎨 Style Examples")
        st.markdown("""
        - "Modern tech product banner with blue gradients"
        - "Retro food advertisement with warm tones"
        - "Minimalist fashion banner with model shot"
        """)

if __name__ == "__main__":
    main()
