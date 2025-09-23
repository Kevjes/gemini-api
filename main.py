import tempfile
from pathlib import Path
import os
import io
from fastapi import FastAPI, HTTPException, Response, UploadFile, File, Form
from pydantic import BaseModel
from dotenv import load_dotenv
from gemini_webapi import GeminiClient

# Charger les variables d'environnement depuis .env
load_dotenv()

# Modèle pour les inputs (prompt texte simple)
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


@app.post("/generate-image-binary")
async def generate_image_binary(prompt_input: Prompt):
    if not client:
        raise HTTPException(status_code=500, detail="Client non initialisé.")

    try:
        full_prompt = f"Generate an image of: {prompt_input.prompt}"
        response = await client.generate_content(full_prompt)

        if not response.images:
            raise HTTPException(status_code=400, detail="Aucune image générée.")

        image = response.images[0]

        # Les images de gemini_webapi sont des objets avec méthode save() async
        # Mais on doit gérer les formats correctement
        try:
            # Tenter de sauvegarder directement
            buffer = io.BytesIO()
            await image.save(buffer)
            image_bytes = buffer.getvalue()
        except Exception as save_error:
            # Si ça échoue, essayer d'accéder aux données brutes
            if hasattr(image, 'data'):
                image_bytes = image.data
            elif hasattr(image, '_data'):
                image_bytes = image._data
            else:
                # Dernier recours : convertir en bytes
                image_bytes = bytes(image)

        return Response(content=image_bytes, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur génération image binaire : {str(e)}")


@app.post("/generate-with-images")
async def generate_with_images(
    prompt: str = Form(...),
    images: list[UploadFile] = File(default=[])
):
    if not client:
        raise HTTPException(status_code=500, detail="Client non initialisé.")

    temp_files = []
    try:
        # Sauvegarder les images en fichiers temporaires
        for img in images:
            if img.filename:  # Vérifier que le fichier existe
                content = await img.read()
                tmp_fd, tmp_path = tempfile.mkstemp(
                    suffix=os.path.splitext(img.filename)[1],
                    dir=tempfile.gettempdir()
                )
                with os.fdopen(tmp_fd, "wb") as f:
                    f.write(content)
                temp_files.append(tmp_path)

        # Appel Gemini avec fichiers
        response = await client.generate_content(
            prompt,
            files=[Path(p) for p in temp_files]
        )

        # Retour image si générée
        if response.images and len(response.images) > 0:
            image = response.images[0]
            try:
                buf = io.BytesIO()
                await image.save(buf)
                buf.seek(0)
                return Response(content=buf.getvalue(), media_type="image/png")
            except Exception as save_error:
                # Si la sauvegarde échoue, essayer d'accéder aux données brutes
                if hasattr(image, 'data'):
                    return Response(content=image.data, media_type="image/png")
                elif hasattr(image, '_data'):
                    return Response(content=image._data, media_type="image/png")
                else:
                    raise HTTPException(status_code=500, detail=f"Impossible de sauvegarder l'image: {str(save_error)}")
        else:
            # Retourner une erreur HTTP au lieu de JSON pour maintenir la cohérence du content-type
            raise HTTPException(status_code=400, detail="Aucune image générée pour ce prompt.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur generate-with-images : {str(e)}")
    finally:
        # Nettoyage des fichiers temporaires
        for path in temp_files:
            try:
                os.remove(path)
            except:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
