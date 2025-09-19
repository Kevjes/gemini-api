import asyncio
import os
import base64
import io
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
from gemini_webapi import GeminiClient

# Charger les variables d'environnement depuis .env
load_dotenv()

# Modèle pour les inputs (prompt)
class Prompt(BaseModel):
    prompt: str

# Modèle pour la réponse texte
class TextResponse(BaseModel):
    text: str

# Modèle pour la réponse image (chemin fichier)
class ImageResponse(BaseModel):
    filename: str
    path: str

# Modèle pour la réponse image en base64
class ImageBase64Response(BaseModel):
    base64_image: str

app = FastAPI(title="Mini Gemini API")

# Client Gemini global
client = None

# Dossier pour sauver les images (pour /generate-image uniquement)
IMAGES_DIR = Path("generated_images")
IMAGES_DIR.mkdir(exist_ok=True)

@app.on_event("startup")
async def startup_event():
    global client
    try:
        # Charger les cookies depuis .env
        psid = os.getenv("GEMINI_PSID")
        psidts = os.getenv("GEMINI_PSIDTS")
        
        if not psid or not psidts:
            raise ValueError("Cookies GEMINI_PSID ou GEMINI_PSIDTS manquants dans .env")
        
        print(f"Cookies chargés : PSID={psid[:10]}..., PSIDTS={psidts[:10]}...")
        
        client = GeminiClient(psid, psidts)
        await client.init(timeout=30, auto_refresh=True)
        print("Client Gemini initialisé !")
    except Exception as e:
        print(f"Erreur init : {e}")
        raise HTTPException(status_code=500, detail=f"Échec init client : {str(e)}")

@app.post("/generate-text", response_model=TextResponse)
async def generate_text(prompt_input: Prompt):
    if not client:
        raise HTTPException(status_code=500, detail="Client non initialisé.")
    
    try:
        response = await client.generate_content(prompt_input.prompt)
        return TextResponse(text=response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur génération : {str(e)}")

@app.post("/generate-image")
async def generate_image(prompt_input: Prompt):
    if not client:
        raise HTTPException(status_code=500, detail="Client non initialisé.")
    
    try:
        full_prompt = f"Generate an image of: {prompt_input.prompt}"
        response = await client.generate_content(full_prompt)
        
        if not response.images:
            raise HTTPException(status_code=400, detail="Aucune image générée. Utilise un prompt clair comme 'a cat in space'.")
        
        image = response.images[0]
        filename = f"image_{int(asyncio.get_event_loop().time())}.png"
        filepath = IMAGES_DIR / filename
        await image.save(path=str(IMAGES_DIR), filename=filename, verbose=False)
        
        return {"filename": filename, "full_path": str(filepath.absolute())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur génération image : {str(e)}")

@app.post("/generate-image-base64", response_model=ImageBase64Response)
async def generate_image_base64(prompt_input: Prompt):
    if not client:
        raise HTTPException(status_code=500, detail="Client non initialisé.")
    
    try:
        full_prompt = f"Generate an image of: {prompt_input.prompt}"
        response = await client.generate_content(full_prompt)
        
        if not response.images:
            raise HTTPException(status_code=400, detail="Aucune image générée.")
        
        image = response.images[0]  # devrait être un PIL.Image
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        image_data = buffer.read()
        
        base64_image = base64.b64encode(image_data).decode("utf-8")
        return ImageBase64Response(base64_image=base64_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur génération image base64 : {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)