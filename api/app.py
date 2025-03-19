import os
import torch
import base64
from io import BytesIO
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
from PIL import Image
import json
import uuid
from datetime import datetime
from typing import List, Optional

app = FastAPI(title="DreamBooth Disney Style API", 
              description="API pour générer des images avec un style Disney via DreamBooth")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À modifier en production pour limiter aux origines autorisées
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèle de requête pour la génération d'image
class ImageRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    negative_prompt: str = None
    style: Optional[str] = None

# Modèle de réponse pour la génération d'image
class ImageResponse(BaseModel):
    image: str  # Base64 encoded image

# Modèle pour un style
class Style(BaseModel):
    id: str
    name: str
    description: str

# Modèle pour une entrée d'historique
class HistoryEntry(BaseModel):
    id: str
    prompt: str
    style: Optional[str] = None
    imageUrl: str
    createdAt: str

# Variables globales
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"
history = []  # Historique des images générées

# Fonction pour charger le modèle
def load_model():
    global model
    if model is None:
        model_path = os.environ.get("MODEL_PATH", "../model")  
        try:
            model = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            model.to(device)
            # Optimisation pour réduire l'utilisation de la mémoire
            if device == "cuda":
                model.enable_attention_slicing()
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            # Fallback au modèle de base si le modèle fine-tuné n'est pas disponible
            model = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            model.to(device)
            if device == "cuda":
                model.enable_attention_slicing()

# Fonction pour convertir une image PIL en base64
def pil_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Fonction pour sauvegarder une entrée dans l'historique
def save_to_history(prompt, style, image_base64):
    entry = {
        "id": str(uuid.uuid4()),
        "prompt": prompt,
        "style": style,
        "imageUrl": f"data:image/png;base64,{image_base64}",
        "createdAt": datetime.now().isoformat()
    }
    history.append(entry)
    
    # Limiter la taille de l'historique (garder les 50 dernières entrées)
    if len(history) > 50:
        history.pop(0)
    
    return entry

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API DreamBooth Disney Style"}

@app.post("/generate", response_model=ImageResponse)
async def generate_image(request: ImageRequest, background_tasks: BackgroundTasks):
    if model is None:
        load_model()
        if model is None:
            raise HTTPException(status_code=500, detail="Impossible de charger le modèle")
    
    try:
        # Préparation du prompt en fonction du style
        prompt = request.prompt
        style = request.style
        
        # Ajout du style au prompt si spécifié
        if style == "sksdisney":
            if "sksdisneystyle" not in prompt:
                prompt = f"{prompt}, sksdisneystyle"
        elif style:
            if f"{style}style" not in prompt:
                prompt = f"{prompt}, {style}style"
        
        # Génération de l'image
        generator = torch.Generator(device=device).manual_seed(42)  # Pour la reproductibilité
        
        # Préparation des paramètres
        params = {
            "prompt": prompt,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "generator": generator
        }
        
        # Ajout du negative_prompt s'il est fourni
        if request.negative_prompt:
            params["negative_prompt"] = request.negative_prompt
        
        # Génération
        result = model(**params)
        
        # Conversion en base64
        image_base64 = pil_to_base64(result.images[0])
        
        # Sauvegarder dans l'historique
        save_to_history(request.prompt, style, image_base64)
        
        return {"image": image_base64}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération: {str(e)}")

@app.post("/transform", response_model=ImageResponse)
async def transform_image(
    image: UploadFile = File(...),
    style: str = Form(...),
    num_inference_steps: int = Form(30),
    guidance_scale: float = Form(7.5)
):
    if model is None:
        load_model()
        if model is None:
            raise HTTPException(status_code=500, detail="Impossible de charger le modèle")
    
    try:
        # Lire et convertir l'image téléchargée
        contents = await image.read()
        input_image = Image.open(BytesIO(contents))
        
        # Redimensionner si nécessaire
        if input_image.width > 512 or input_image.height > 512:
            input_image.thumbnail((512, 512))
        
        # Préparation du prompt en fonction du style
        prompt = f"Transform this image in {style} style"
        
        # Génération de l'image
        generator = torch.Generator(device=device).manual_seed(42)
        
        # Préparation des paramètres (ici on simule la transformation)
        # Note: Dans une implémentation réelle, vous utiliseriez un modèle de transformation d'image
        params = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator
        }
        
        # Génération (simulation)
        result = model(**params)
        
        # Conversion en base64
        image_base64 = pil_to_base64(result.images[0])
        
        # Sauvegarder dans l'historique
        save_to_history("Image transformée", style, image_base64)
        
        return {"image": image_base64}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la transformation: {str(e)}")

@app.get("/styles", response_model=List[Style])
async def get_styles():
    # Liste des styles disponibles
    styles = [
        {"id": "sksdisney", "name": "Disney", "description": "Style inspiré des films Disney"},
        {"id": "anime", "name": "Anime", "description": "Style manga japonais"},
        {"id": "pixar", "name": "Pixar", "description": "Style des films Pixar"},
        {"id": "watercolor", "name": "Aquarelle", "description": "Style peinture à l'aquarelle"},
        {"id": "comic", "name": "Comic", "description": "Style bande dessinée"}
    ]
    return styles

@app.get("/history", response_model=List[HistoryEntry])
async def get_history():
    # Retourner l'historique des images générées
    return history

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# Point d'entrée pour le démarrage avec Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)