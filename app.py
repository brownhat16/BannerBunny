import streamlit as st
import requests
import base64
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
    st.title("🎨 Banner Generation UI")
    st.caption("Powered by BannerBunny API")

    with st.sidebar:
        st.header("Generation Parameters")
        prompt = st.text_area("Banner Prompt", help="Describe the banner you want to generate")
        width = st.number_input("Width", 512, 2048, 1024)
        height = st.number_input("Height", 512, 2048, 768)
        seed = st.number_input("Seed (optional)", min_value=0, value=None)
        async_mode = st.checkbox("Async Processing", value=False)
    
    if st.button("Generate Banner"):
        if not prompt:
            st.error("Please enter a banner prompt")
            return

        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "seed": seed or None
        }

        with st.spinner("🚀 Generating banner..."):
            result = generate_banner(payload, async_mode)
            
            if "error" in result:
                st.error(f"API Error: {result['error']}")
                return

            if async_mode:
                request_id = result.get("request_id")
                while True:
                    status = poll_async_status(request_id)
                    if status.get("status") == "completed":
                        result = status.get("result")
                        break
                    if status.get("status") == "failed":
                        st.error("Async generation failed")
                        return
                    st.write(f"Status: {status.get('progress')}")
                    time.sleep(2)

            if result.get("image_base64"):
                col1, col2 = st.columns(2)
                with col1:
                    image_data = base64.b64decode(result["image_base64"])
                    image = Image.open(BytesIO(image_data))
                    st.image(image, caption="Generated Banner", use_column_width=True)
                    
                    # Download button
                    buf = BytesIO()
                    image.save(buf, format="PNG")
                    st.download_button(
                        label="Download Banner",
                        data=buf.getvalue(),
                        file_name="generated_banner.png",
                        mime="image/png"
                    )
                
                with col2:
                    st.subheader("Generation Details")
                    st.json({
                        "Processing Time": f"{result.get('processing_time', 0):.2f}s",
                        "Request ID": result.get("request_id"),
                        "Flux Prompt": result.get("flux_prompt")
                    })
                    
                    with st.expander("Structured Metadata"):
                        st.json(result.get("structured_data", {}))
            else:
                st.error("Banner generation failed")

if __name__ == "__main__":
    main()
