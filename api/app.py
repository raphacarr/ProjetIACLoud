import os
import torch
import base64
from io import BytesIO
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
from PIL import Image

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

# Modèle de réponse pour la génération d'image
class ImageResponse(BaseModel):
    image: str  # Base64 encoded image

# Variables globales
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

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
        # Ajout du style Disney au prompt si pas déjà présent
        if "sksdisneystyle" not in request.prompt:
            prompt = f"{request.prompt}, sksdisneystyle"
        else:
            prompt = request.prompt
        
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
        
        return {"image": image_base64}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# Point d'entrée pour le démarrage avec Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)