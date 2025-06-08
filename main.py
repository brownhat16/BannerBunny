import json
import requests
from typing import Dict, Any, Optional, List
import time
import logging
import asyncio
import aiohttp
import ssl
import certifi
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from io import BytesIO
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Pydantic Models
class BannerRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 768
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    seed: Optional[int] = None


class BannerResponse(BaseModel):
    status: str
    request_id: str
    structured_data: Optional[Dict] = None
    flux_prompt: Optional[str] = None
    image_base64: Optional[str] = None
    processing_time: float
    error: Optional[str] = None


class StatusResponse(BaseModel):
    status: str
    message: str
    timestamp: str


# Global storage for async jobs
processing_jobs = {}


class CompleteBannerPipeline:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.together.xyz/v1/chat/completions"
        self.image_generation_url = "https://api.together.xyz/v1/images/generations"
        self.model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
        self.image_model = "black-forest-labs/FLUX.1-dev"

        # Prompt descriptors
        self.color_descriptors = {
            "Red": "vibrant red", "Green": "festive green", "Blue": "deep blue",
            "Yellow": "bright yellow", "Orange": "warm orange", "Purple": "rich purple",
            "Pink": "soft pink", "White": "clean white", "Black": "bold black",
            "Brown": "warm brown", "Grey": "neutral grey"
        }
        self.mood_descriptors = {
            "Calm": "serene and peaceful", "Energetic": "dynamic and vibrant",
            "Luxurious": "elegant and premium", "Playful": "fun and engaging",
            "Professional": "clean and corporate", "Cozy": "warm and inviting"
        }
        self.theme_descriptors = {
            "Modern": "contemporary and sleek", "Classic": "timeless and traditional",
            "Retro": "vintage-inspired", "Minimalist": "clean and uncluttered",
            "Corporate": "professional and structured", "Festive": "celebratory and joyful"
        }

    def create_system_prompt(self) -> str:
        return """You are an AI assistant that converts user requests for advertising banners into a structured JSON format. Analyze the user's prompt and extract the relevant attributes according to the provided schema. For any attribute not mentioned in the user's prompt, use a sensible default value from the allowed options.
You must respond with ONLY a valid JSON object that matches this exact schema based on the provided metadata table:
{
  "Dominant colors": "Red/Yellow/Blue/Green/Orange/Purple/Pink/White/Black/Brown/Grey",
  "Brightness": "Light/Dark/Medium",
  "Warm vs cool tones": "Warm/Cool/Neutral",
  "Contrast level": "High/Medium/Low",
  "Text-to-image ratio": "10%/30%/50%/70%/90%",
  "Left vs right alignment": "Left/Right/Center",
  "Symmetry": "Symmetrical/Asymmetrical",
  "Whitespace usage": "Low/Medium/High",
  "Image focus type": "Product/Lifestyle",
  "Visual format": "Static/Animated",
  "Number vs photo": "Number/Photo",
  "Number of products shown": "1/2/3+",
  "Number of people shown": "0/1/2/3+",
  "Design density": "Minimal/Medium/Dense",
  "Embedded text present": "Yes/No",
  "Text language": "English/Hindi/Others",
  "Font style": "Bold/Serif/Sans-serif",
  "Festival special occasion logo": "Yes/No",
  "Festival name": "Diwali/Holi/Christmas/Eid/New Year/Valentine/Mother's Day/Father's Day/Independence Day/Republic Day/None",
  "Logo size": "Small/Medium/Large",
  "Logo placement": "Top Left/Top Center/Top Right/Center Left/Center/Center Right/Bottom Left/Bottom Center/Bottom Right",
  "Call-to-action button present": "Yes/No",
  "CTA placement": "Top/Center/Bottom",
  "CTA contrast": "High/Medium/Low",
  "CTA text": "string",
  "Objects visible": "Yes/No",
  "Brand logo visible": "Yes/No",
  "Brand name size": "Small/Medium/Large",
  "Emotion (if faces shown)": "Happy/Excited/Calm/Serious/Surprised/Sad/Angry/None",
  "Gender shown (if people shown)": "Male/Female/Mixed/None",
  "Employment type (if shown)": "Indoor/Outdoor/None",
  "Environment type": "Indoor/Outdoor/None",
  "Location type": "Kitchen/Living room/Bedroom/Office/Store/Street/Park/Beach/Mountain/Other/None",
  "Offer text present": "Yes/No",
  "Offers": "Kitchen/Electronics/Fashion/Home/Beauty/Sports/Books/Travel/Food/Other/None",
  "Offer text size": "Small/Medium/Large/None",
  "Offer text position": "Top Left/Top Center/Top Right/Center Left/Center/Center Right/Bottom Left/Bottom Center/Bottom Right/None",
  "Offer text content": "string/None",
  "Key points / Highlights": ["string"],
  "Elements placement logic": "Grid Layout/Freeform/Overlapping/Stacked",
  "Person positioning (if shown)": "Front Facing/Side Profile/Group Shot/Not Applicable",
  "Banner layout orientation": "Horizontal/Vertical/Square",
  "Accessibility features": "High Contrast/Alt Text/Readable Fonts/None",
  "Theme": "Modern/Classic/Retro/Minimalist/Corporate/Festive",
  "Tone & Mood": "Energetic/Calm/Luxurious/Playful/Professional/Cozy",
  "Brand tagline": "string/Not Applicable",
  "Background texture": "Solid/Gradient/Pattern/Photographic/Abstract"
}
Rules:
1. Respond with ONLY the JSON object, no additional text or explanations
2. Use exact field names and values as specified above
3. If a value is not mentioned in the prompt, choose the most appropriate default"""

    async def extract_metadata_async(self, user_prompt: str, max_retries: int = 3) -> Optional[Dict]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.create_system_prompt()},
                {"role": "user", "content": f"Convert this banner request to JSON: {user_prompt}"}
            ],
            "max_tokens": 1500,
            "temperature": 0.1,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }

        logger.info(f"Extracting metadata for prompt: {user_prompt[:100]}...")

        ssl_context = ssl.create_default_context(cafile=certifi.where())

        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
                    async with session.post(
                            self.base_url,
                            headers=headers,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=60)
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"API Error {response.status}: {error_text}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            return None
                        result = await response.json()
                        if "error" in result:
                            logger.error(f"API Error: {result['error']}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            return None
                        assistant_message = result["choices"][0]["message"]["content"].strip()
                        cleaned_message = self._clean_json_response(assistant_message)

                        try:
                            metadata = json.loads(cleaned_message)
                            logger.info("✅ Metadata extraction successful")
                            return metadata
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse JSON: {e}")
                            return None
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
        logger.error("❌ All metadata extraction attempts failed")
        return None

    def _clean_json_response(self, response: str) -> str:
        response = response.strip()

        if response.startswith("```json"):
            response = response[5:]
        elif response.startswith("```"):
            response = response[3:]

        if response.endswith("```"):
            response = response[:-3]

        response = response.strip()

        start_idx = response.find('{')
        end_idx = response.rfind('}')

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            response = response[start_idx:end_idx + 1]
        return response

    def convert_to_flux_prompt(self, metadata: Dict[str, Any]) -> str:
        prompt_parts = []

        orientation = metadata.get("Banner layout orientation", "Horizontal").lower()
        banner_type = f"Professional {orientation} advertising banner"
        prompt_parts.append(banner_type)

        theme = metadata.get("Theme", "Modern")
        mood = metadata.get("Tone & Mood", "Professional")
        theme_desc = self.theme_descriptors.get(theme, theme.lower())
        mood_desc = self.mood_descriptors.get(mood, mood.lower())
        prompt_parts.append(f"with {theme_desc} design aesthetic, {mood_desc} atmosphere")

        colors = metadata.get("Dominant colors", "Blue").split("/")
        color_desc = " and ".join([self.color_descriptors.get(c.strip(), c.strip().lower()) for c in colors])
        brightness = metadata.get("Brightness", "Medium").lower()
        tone_temp = metadata.get("Warm vs cool tones", "Neutral").lower()
        prompt_parts.append(f"featuring {color_desc} color palette, {brightness} brightness, {tone_temp} tones")

        density = metadata.get("Design density", "Medium").lower()
        whitespace = metadata.get("Whitespace usage", "Medium").lower()
        alignment = metadata.get("Left vs right alignment", "Center").lower()
        symmetry = metadata.get("Symmetry", "Symmetrical").lower()
        placement_logic = metadata.get("Elements placement logic", "Grid Layout").lower()
        composition_desc = f"{density} design density with {whitespace} whitespace usage, {alignment} aligned elements in {symmetry} {placement_logic}"
        prompt_parts.append(composition_desc)

        focus_type = metadata.get("Image focus type", "Product").lower()
        num_products = metadata.get("Number of products shown", "1")
        if focus_type == "product" and num_products != "0":
            product_desc = self._build_product_description(metadata)
            prompt_parts.append(product_desc)

        text_elements = self._build_text_description(metadata)
        if text_elements:
            prompt_parts.append(text_elements)

        environment = self._build_environment_description(metadata)
        if environment:
            prompt_parts.append(environment)

        tech_specs = self._build_technical_specs(metadata)
        if tech_specs:
            prompt_parts.append(tech_specs)

        quality_enhancers = [
            "high-quality commercial photography",
            "professional marketing design",
            "clean and polished finish",
            "sharp details and crisp edges",
            "optimized for digital display"
        ]

        complete_prompt = ", ".join(prompt_parts + quality_enhancers)
        logger.info(f"Generated FLUX prompt: {complete_prompt[:100]}...")
        return complete_prompt

    def _build_product_description(self, metadata: Dict[str, Any]) -> str:
        num_products = metadata.get("Number of products shown", "1")
        offer_category = metadata.get("Offers", "")
        product_terms = {
            "Electronics": "sleek electronic devices", "Fashion": "stylish clothing items",
            "Home": "elegant home furnishings", "Beauty": "premium beauty products",
            "Sports": "athletic equipment", "Kitchen": "modern kitchen appliances",
            "Travel": "travel accessories", "Food": "gourmet food items"
        }
        if offer_category in product_terms:
            product_desc = product_terms[offer_category]
        else:
            product_desc = "premium products"
        if num_products == "1":
            return f"showcasing one featured {product_desc}"
        elif num_products == "2":
            return f"displaying two {product_desc} elegantly arranged"
        else:
            return f"featuring multiple {product_desc} in an organized display"

    def _build_text_description(self, metadata: Dict[str, Any]) -> str:
        text_elements = []

        if metadata.get("Offer text present") == "Yes":
            offer_content = metadata.get("Offer text content", "Special Offer")
            offer_size = metadata.get("Offer text size", "Medium").lower()
            offer_position = metadata.get("Offer text position", "Top Center").lower()
            text_elements.append(f"prominent {offer_size} '{offer_content}' text positioned at {offer_position}")

        if metadata.get("Call-to-action button present") == "Yes":
            cta_text = metadata.get("CTA text", "Shop Now")
            cta_placement = metadata.get("CTA placement", "Bottom").lower()
            cta_contrast = metadata.get("CTA contrast", "High").lower()
            text_elements.append(f"{cta_contrast} contrast '{cta_text}' button positioned at {cta_placement}")

        font_style = metadata.get("Font style", "Sans-serif").lower()
        if font_style:
            text_elements.append(f"using clean {font_style} typography")
        return ", ".join(text_elements) if text_elements else ""

    def _build_environment_description(self, metadata: Dict[str, Any]) -> str:
        env_elements = []

        festival = metadata.get("Festival name", "")
        if festival and festival != "None":
            env_elements.append(f"with {festival} themed decorative elements")

        location = metadata.get("Location type", "")
        if location and location not in ["Other", "None"]:
            env_elements.append(f"set in {location.lower()} environment")

        background = metadata.get("Background texture", "Solid").lower()
        if background != "solid":
            env_elements.append(f"with {background} background texture")

        return ", ".join(env_elements) if env_elements else ""

    def _build_technical_specs(self, metadata: Dict[str, Any]) -> str:
        specs = []
        contrast = metadata.get("Contrast level", "Medium").lower()
        specs.append(f"{contrast} contrast ratio")
        accessibility = metadata.get("Accessibility features", "")
        if accessibility and accessibility != "None":
            specs.append("accessible design with readable fonts")
        visual_format = metadata.get("Visual format", "Static").lower()
        specs.append(f"{visual_format} banner format")
        return ", ".join(specs) if specs else ""

    async def generate_image_async(self, prompt: str, width: int = 1024, height: int = 768,
                                 num_inference_steps: int = 28, guidance_scale: float = 3.5,
                                 seed: Optional[int] = None) -> Optional[str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.image_model,
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": num_inference_steps,
            "n": 1,
            "response_format": "b64_json"
        }
        if seed is not None:
            payload["seed"] = seed

        logger.info(f"Generating image with FLUX.1-dev: {width}x{height}")

        ssl_context = ssl.create_default_context(cafile=certifi.where())

        try:
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
                async with session.post(
                        self.image_generation_url,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Image generation failed {response.status}: {error_text}")
                        return None
                    result = await response.json()
                    if "error" in result:
                        logger.error(f"Image generation error: {result['error']}")
                        return None
                    if "data" in result and len(result["data"]) > 0:
                        image_b64 = result["data"][0]["b64_json"]
                        logger.info("✅ Image generation successful")
                        return image_b64
                    else:
                        logger.error("No image data returned")
                        return None
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return None

    async def process_banner_request_async(self, user_prompt: str, width: int = 1024,
                                         height: int = 768, num_inference_steps: int = 28,
                                         guidance_scale: float = 3.5, seed: Optional[int] = None) -> Dict:
        start_time = time.time()
        logger.info(f"🔄 Processing banner request: {user_prompt[:100]}...")

        metadata = await self.extract_metadata_async(user_prompt)
        if not metadata:
            return {
                "status": "error",
                "error": "Failed to extract metadata",
                "processing_time": time.time() - start_time
            }

        flux_prompt = self.convert_to_flux_prompt(metadata)

        image_b64 = await self.generate_image_async(
            flux_prompt, width, height, num_inference_steps, guidance_scale, seed
        )

        processing_time = time.time() - start_time

        if image_b64:
            logger.info(f"✅ Complete pipeline successful in {processing_time:.2f}s")
            return {
                "status": "success",
                "structured_data": metadata,
                "flux_prompt": flux_prompt,
                "image_base64": image_b64,
                "processing_time": processing_time
            }
        else:
            logger.error("❌ Image generation failed")
            return {
                "status": "partial_success",
                "structured_data": metadata,
                "flux_prompt": flux_prompt,
                "error": "Image generation failed",
                "processing_time": processing_time
            }


# Initialize Pipeline
API_KEY = "50f7427ca836843296c3ceccd2092c504ca45f20e30cb922e0c35cfb7046aeb0"
pipeline = CompleteBannerPipeline(API_KEY)


# FastAPI App
app = FastAPI(
    title="Banner Generation API",
    description="Generate advertising banners from natural language prompts",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=StatusResponse)
async def root():
    return StatusResponse(
        status="online",
        message="Banner Generation API is running",
        timestamp=datetime.now().isoformat()
    )


@app.post("/generate-banner", response_model=BannerResponse)
async def generate_banner(request: BannerRequest):
    request_id = f"req_{int(time.time() * 1000)}"
    start_time = time.time()
    logger.info(f"📨 Received banner request {request_id}: {request.prompt}")

    try:
        result = await pipeline.process_banner_request_async(
            user_prompt=request.prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed
        )
        processing_time = time.time() - start_time
        return BannerResponse(
            status=result["status"],
            request_id=request_id,
            structured_data=result.get("structured_data"),
            flux_prompt=result.get("flux_prompt"),
            image_base64=result.get("image_base64"),
            processing_time=processing_time,
            error=result.get("error")
        )
    except Exception as e:
        logger.error(f"❌ Request {request_id} failed: {str(e)}")
        return BannerResponse(
            status="error",
            request_id=request_id,
            processing_time=time.time() - start_time,
            error=str(e)
        )


@app.post("/generate-banner-async")
async def generate_banner_async(request: BannerRequest, background_tasks: BackgroundTasks):
    request_id = f"async_req_{int(time.time() * 1000)}"
    logger.info(f"📨 Received async banner request {request_id}: {request.prompt}")
    processing_jobs[request_id] = {
        "status": "processing",
        "progress": "Starting...",
        "created_at": datetime.now().isoformat(),
        "result": None
    }
    background_tasks.add_task(
        process_async_banner,
        request_id,
        request.prompt,
        request.width,
        request.height,
        request.num_inference_steps,
        request.guidance_scale,
        request.seed
    )
    return {
        "request_id": request_id,
        "status": "accepted",
        "message": "Banner generation started",
        "check_status_url": f"/status/{request_id}"
    }


@app.get("/status/{request_id}")
async def get_job_status(request_id: str):
    if request_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = processing_jobs[request_id]
    return {
        "request_id": request_id,
        "status": job["status"],
        "progress": job["progress"],
        "created_at": job["created_at"],
        "result": job["result"]
    }


@app.get("/image/{request_id}")
async def get_generated_image(request_id: str):
    if request_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = processing_jobs[request_id]
    if job["status"] != "completed" or not job["result"] or not job["result"].get("image_base64"):
        raise HTTPException(status_code=404, detail="Image not available")
    try:
        image_data = base64.b64decode(job["result"]["image_base64"])
        image = Image.open(BytesIO(image_data))
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format=image.format)
        img_byte_arr.seek(0)
        return StreamingResponse(
            BytesIO(img_byte_arr.getvalue()),
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=banner_{request_id}.png"}
        )
    except Exception as e:
        logger.error(f"Error serving image {request_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


async def process_async_banner(request_id: str, prompt: str, width: int, height: int,
                              num_inference_steps: int, guidance_scale: float, seed: Optional[int]):
    try:
        result = await pipeline.process_banner_request_async(
            user_prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        processing_jobs[request_id].update({
            "status": "completed",
            "progress": "Complete",
            "result": result,
            "completed_at": datetime.now().isoformat()
        })
        logger.info(f"✅ Async request {request_id} completed successfully")
    except Exception as e:
        processing_jobs[request_id].update({
            "status": "failed",
            "progress": f"Error: {str(e)}",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })
        logger.error(f"❌ Async request {request_id} failed: {str(e)}")


# Run Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
