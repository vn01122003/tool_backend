import os
import base64
import re
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from rembg import remove

# Load .env file
load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_KEY")
QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
FLUX_MODEL_ID = os.getenv("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-dev")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def step_remove_background(image_bytes: bytes) -> bytes:
    return remove(image_bytes)

# Step 1: Analyze image with Qwen2.5-VL
def step_analyze_image(image_bytes: bytes) -> str:
    client = InferenceClient(model=QWEN_MODEL, token=HF_API_TOKEN)
    image_b64 = base64.b64encode(image_bytes).decode()

    response = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image as accurately as possible: include exact layout, all visible objects, characters, and text (with position, font style, and effects like bold or distressed). Note colors, background, and visual style. Avoid any interpretation — describe only what is clearly seen, from top to bottom, left to right. The description must be purely factual and no longer than 700 characters."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                ]
            }
        ],
    )
    return response.choices[0].message.content

# Step 2: Generate art prompt
def step_generate_prompt(caption: str) -> str:
    system_prompt = """
    You are a prompt generation expert for T-shirt art. Write a prompt for a text-to-image AI (e.g., FLUX.1) to accurately reconstruct a design from a user description. Preserve:
    1. Exact layout and alignment of elements,
    2. Structure and pose of any characters or objects,
    3. Text content, font style, size, effects (distressed, curved),
    4. Color scheme, background (transparent or textured),
    5. Print-like textures and outlines,
    6. 2D flat look, no added depth or stylization.
    Enhance resolution only if it improves fidelity.
    Return a single plain-text sentence (no markdown) under 1999 characters.
    """
    final_prompt = f"{system_prompt.strip()} User description: {caption.strip()}"
    return final_prompt

# Step 3: Escape prompt
def step_escape_prompt(prompt: str) -> str:
    cleaned = prompt.replace("\n", " ").strip()
    escaped = cleaned.replace("\\", "\\\\").replace('"', '\\"')
    if escaped.startswith('\\"'): escaped = escaped[2:]
    if escaped.endswith('\\"'): escaped = escaped[:-2]
    return escaped

# Step 4: Generate image with FLUX.1
def step_generate_image(prompt: str) -> BytesIO:
    client = InferenceClient(provider="nebius", api_key=HF_API_TOKEN)
    image = client.text_to_image(
        prompt=prompt,
        model=FLUX_MODEL_ID,
        guidance_scale=7.5,
        num_inference_steps=50
    )
    output_buffer = BytesIO()
    image.save(output_buffer, format="PNG")
    output_buffer.seek(0)
    return output_buffer

@app.get("/")
def read_root():
    return {"message": "Backend is running!"}

@app.post("/webhook")
async def generate_from_image(request: Request):
    try:
        data = await request.json()
        base64_image = data.get("chatInput")
        if not base64_image:
            print("❌ No 'chatInput' found in request.")
            return {"error": "Missing 'chatInput'"}

        print("📥 Received base64 image from extension.")

        # Clean base64 string
        base64_clean = re.sub(r"^data:image\/\w+;base64,", "", base64_image)
        image_bytes = base64.b64decode(base64_clean)

        # Remove background before captioning
        print("🧼 Step 0: Removing background...")
        image_bytes_no_bg = step_remove_background(image_bytes)


        print("🔍 Step 1: Analyzing image with Qwen2.5-VL...")
        caption = step_analyze_image(image_bytes_no_bg)
        print(f"✅ Caption received:\n{caption}")

        print("📝 Step 2: Generating prompt...")
        prompt = step_generate_prompt(caption)
        print(f"🧾 Prompt:\n{prompt}")

        print("🚿 Step 3: Escaping prompt...")
        escaped = step_escape_prompt(prompt)
        print(f"🔐 Escaped Prompt:\n{escaped}")

        print("🎨 Step 4: Generating image with FLUX.1...")
        final_image = step_generate_image(escaped)
        print("✅ Image generation complete.")

        return StreamingResponse(final_image, media_type="image/png")

    except Exception as e:
        print("❌ Error occurred:", str(e))
        return {"error": str(e)}
