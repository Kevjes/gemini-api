import asyncio
import os
import base64
import io
from fastapi import FastAPI, HTTPException, Response
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

app = FastAPI(title="Mini Gemini API")

# Client Gemini global
client = None

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

# Nouvelle route : renvoie directement l'image binaire
@app.post("/generate-image-binary")
async def generate_image_binary(prompt_input: Prompt):
    if not client:
        raise HTTPException(status_code=500, detail="Client non initialisé.")
    
    try:
        full_prompt = f"Generate an image of: {prompt_input.prompt}"
        response = await client.generate_content(full_prompt)
        
        if not response.images:
            raise HTTPException(status_code=400, detail="Aucune image générée.")
        
        # Récupérer l'objet image (PIL.Image ou bytes-like selon gemini_webapi)
        image = response.images[0]

        # Vérifier si c'est un objet PIL.Image
        if hasattr(image, "save"):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
        else:
            # Si c'est déjà un buffer/bytes renvoyé par gemini_webapi
            image_bytes = bytes(image)

        return Response(content=image_bytes, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur génération image binaire : {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
