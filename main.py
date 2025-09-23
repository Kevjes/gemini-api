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

        # Accéder directement aux données binaires selon la documentation officielle
        if hasattr(image, 'bytes') and image.bytes:
            # Méthode recommandée : accès direct aux bytes
            image_bytes = image.bytes
        elif hasattr(image, 'url') and image.url:
            # Fallback : télécharger depuis l'URL avec configuration Google
            import httpx
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            async with httpx.AsyncClient(
                follow_redirects=True,
                max_redirects=50,
                timeout=120.0,
                headers=headers
            ) as http_client:
                response = await http_client.get(image.url)
                image_bytes = response.content
        else:
            # Dernier recours pour compatibilité
            raise HTTPException(status_code=500, detail="Image sans données bytes ni URL accessible")

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

        # Appel Gemini avec retry automatique pour gérer l'instabilité de l'IA
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await client.generate_content(
                    prompt,
                    files=[Path(p) for p in temp_files]
                )

                # Retour image si générée
                if response.images and len(response.images) > 0:
                    image = response.images[0]

                    # Accéder directement aux données binaires selon la documentation officielle
                    if hasattr(image, 'bytes') and image.bytes:
                        # Méthode recommandée : accès direct aux bytes
                        return Response(content=image.bytes, media_type="image/png")
                    elif hasattr(image, 'url') and image.url:
                        # Fallback : télécharger depuis l'URL avec configuration Google
                        import httpx
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                        }
                        async with httpx.AsyncClient(
                            follow_redirects=True,
                            max_redirects=50,
                            timeout=120.0,
                            headers=headers
                        ) as http_client:
                            img_response = await http_client.get(image.url)
                            return Response(content=img_response.content, media_type="image/png")
                    else:
                        # Aucune donnée image accessible
                        raise HTTPException(status_code=500, detail="Image générée mais données inaccessibles")
                else:
                    # Aucune image générée - instabilité de l'IA, retry automatique
                    if attempt < max_retries - 1:
                        print(f"Tentative {attempt + 1}/{max_retries} - Aucune image générée, retry...")
                        continue
                    else:
                        # Dernier essai échoué
                        raise HTTPException(status_code=400, detail="Aucune image générée après 3 tentatives.")

            except HTTPException:
                # Re-lever les HTTPException (comme les erreurs 400/500)
                raise
            except Exception as e:
                # Autres erreurs : retry si ce n'est pas le dernier essai
                if attempt < max_retries - 1:
                    print(f"Tentative {attempt + 1}/{max_retries} - Erreur: {e}, retry...")
                    continue
                else:
                    raise HTTPException(status_code=500, detail=f"Erreur après 3 tentatives: {str(e)}")

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
