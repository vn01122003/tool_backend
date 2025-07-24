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
    You are a prompt generation expert specializing in highly accurate image reconstruction with subtle enhancement. 
    Your goal is to generate a prompt for a text-to-image model (e.g., FLUX.1) that faithfully reproduces the original image, with only minimal artistic change.
    Requirements:
    1. Retain exact layout, positioning, and proportions of all objects.
    2. Preserve original color tones and material textures.
    3. Match the original illustration or photo style. Slight improvement is allowed, but avoid stylization.
    4. You may enhance resolution, edge clarity, and lighting to make the image more vivid or realistic, 
    but visual structure and identity must remain within 30% deviation from the original.
    5. If the original includes a hook, tag, or mounting string — these must remain unchanged.
    6. The background should be removed or made clean and neutral unless otherwise instructed.
    7. The output must feel like a higher-quality version of the original, not a stylistic reinterpretation.
    Return a single sentence prompt, under 1999 characters, describing the image with these rules in mind.
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
        extra_prompt = data.get("extraPrompt", "")  # ✅ NEW: get extraPrompt from request

        if not base64_image:
            print("❌ No 'chatInput' found in request.")
            return {"error": "Missing 'chatInput'"}

        print("📥 Received base64 image from extension.")

        # Clean base64 string
        base64_clean = re.sub(r"^data:image\/\w+;base64,", "", base64_image)
        image_bytes = base64.b64decode(base64_clean)

        print("🔍 Step 1: Analyzing image with Qwen2.5-VL...")
        caption = step_analyze_image(image_bytes)

        # ✅ NEW: Append user-controlled prompt logic
        if extra_prompt:
            caption += " " + extra_prompt

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
