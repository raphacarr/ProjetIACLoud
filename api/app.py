import os
import torch
import base64
import boto3
import shutil
import tempfile
from io import BytesIO
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
from PIL import Image
import json
import uuid
from datetime import datetime
from typing import List, Optional, Dict

app = FastAPI(title="API projet IA Cloud - DreamGenerator", 
              description="API pour générer des images avec différents styles via DreamBooth")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.environ.get("FRONTEND_URL", "*"),
        "https://mvrmruieji.eu-west-3.awsapprunner.com",  # URL spécifique de votre frontend
        "http://localhost:3000"  # Pour le développement local
    ],
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
models = {}  # Dictionnaire pour stocker les différents modèles
device = "cuda" if torch.cuda.is_available() else "cpu"
history = []  # Historique des images générées

# Configuration S3
S3_BUCKET = "ia-cloud-models"
S3_DISNEY_MODEL_PATH = "disney"
LOCAL_MODELS_DIR = os.environ.get("MODEL_PATH", "../model")
# Fonction pour télécharger un modèle depuis S3
def download_model_from_s3(bucket_name, s3_path, local_path):
    try:
        print(f"Téléchargement du modèle depuis S3: {bucket_name}/{s3_path} vers {local_path}")
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        
        # Créer le répertoire local si nécessaire
        os.makedirs(local_path, exist_ok=True)
        
        # Télécharger tous les fichiers du modèle
        for obj in bucket.objects.filter(Prefix=s3_path):
            # Créer le chemin local pour le fichier
            local_file_path = os.path.join(local_path, os.path.relpath(obj.key, s3_path))
            # Créer les sous-répertoires si nécessaire
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            # Télécharger le fichier
            bucket.download_file(obj.key, local_file_path)
            
        print(f"Modèle téléchargé avec succès dans {local_path}")
        return True
    except Exception as e:
        print(f"Erreur lors du téléchargement depuis S3: {str(e)}")
        return False

# Fonction pour charger le modèle de base
def load_base_model():
    try:
        model_path = os.path.join(LOCAL_MODELS_DIR, "base")
        print(f"Chargement du modèle de base depuis {model_path}...")
        
        # Vérifier si le modèle existe, sinon le télécharger
        if not os.path.exists(model_path) or len(os.listdir(model_path)) == 0:
            print("Modèle de base non trouvé, téléchargement depuis Hugging Face...")
            # Créer le répertoire si nécessaire
            os.makedirs(model_path, exist_ok=True)
            
            # Télécharger le modèle depuis Hugging Face
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="runwayml/stable-diffusion-v1-5",  # Modèle de base Stable Diffusion
                local_dir=model_path,
                ignore_patterns=["*.safetensors", "*.bin", "*.onnx"]  # Télécharger uniquement les fichiers nécessaires
            )
            print(f"Modèle de base téléchargé avec succès dans {model_path}")
        
        # Optimisations pour réduire l'utilisation de la mémoire
        model = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,  # Désactiver le safety checker pour économiser de la mémoire
            requires_safety_checker=False
        )
        
        # Si CUDA est disponible, utiliser la moitié de précision et déplacer sur GPU
        if torch.cuda.is_available():
            model = model.to("cuda")
            print("Modèle de base chargé sur GPU avec succès")
        else:
            # Optimisations pour CPU
            model.enable_attention_slicing()
            print("Modèle de base chargé sur CPU avec succès (attention slicing activé)")
            
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle de base: {str(e)}")
        return None

# Fonction pour charger le modèle Disney
def load_disney_model():
    try:
        model_path = os.path.join(LOCAL_MODELS_DIR, "disney")
        print(f"Chargement du modèle Disney depuis {model_path}...")
        
        # Vérifier si le modèle existe, sinon le télécharger depuis S3
        if not os.path.exists(model_path) or len(os.listdir(model_path)) == 0:
            print("Modèle Disney non trouvé, téléchargement depuis S3...")
            success = download_model_from_s3(S3_BUCKET, S3_DISNEY_MODEL_PATH, model_path)
            if not success:
                print("Échec du téléchargement du modèle Disney depuis S3, utilisation du modèle de base à la place")
                return load_base_model()
        
        # Optimisations pour réduire l'utilisation de la mémoire
        model = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,  # Désactiver le safety checker pour économiser de la mémoire
            requires_safety_checker=False
        )
        
        # Si CUDA est disponible, utiliser la moitié de précision et déplacer sur GPU
        if torch.cuda.is_available():
            model = model.to("cuda")
            print("Modèle Disney chargé sur GPU avec succès")
        else:
            # Optimisations pour CPU
            model.enable_attention_slicing()
            print("Modèle Disney chargé sur CPU avec succès (attention slicing activé)")
            
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle Disney: {str(e)}")
        print("Utilisation du modèle de base à la place")
        return load_base_model()

# Fonction pour obtenir le modèle approprié en fonction du style
def get_model_for_style(style):
    global models
    
    # Si le style est Disney, utiliser le modèle Disney
    if style == "sksdisney" or style == "disney":
        if "disney" not in models:
            print("Chargement du modèle Disney...")
            models["disney"] = load_disney_model()
        return models["disney"]
    else:
        # Pour tous les autres styles, utiliser le modèle de base
        if "base" not in models:
            print("Chargement du modèle de base...")
            models["base"] = load_base_model()
        return models["base"]

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
    # Chargement des modèles en arrière-plan pour ne pas bloquer le démarrage
    import threading
    threading.Thread(target=lambda: get_model_for_style("base")).start()
    print("Démarrage de l'API - Chargement du modèle de base en arrière-plan")
    
    # Créer le répertoire des modèles s'il n'existe pas
    os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.post("/generate", response_model=ImageResponse)
async def generate_image(request: ImageRequest, background_tasks: BackgroundTasks):
    try:
        # Préparation du prompt en fonction du style
        prompt = request.prompt
        style = request.style
        
        # Sélection du modèle approprié en fonction du style
        selected_model = get_model_for_style(style)
        if selected_model is None:
            raise HTTPException(status_code=500, detail="Impossible de charger le modèle")
        
        # Ajout du style au prompt si spécifié et si on n'utilise pas le modèle Disney spécifique
        # Pour le modèle Disney, pas besoin d'ajouter le style au prompt car le modèle est déjà entraîné
        if style == "sksdisney" and "disney" not in models:
            if "sksdisneystyle" not in prompt:
                prompt = f"{prompt}, sksdisneystyle"
        elif style and style != "sksdisney":
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
        
        # Génération avec le modèle sélectionné
        result = selected_model(**params)
        
        # Conversion en base64
        image_base64 = pil_to_base64(result.images[0])
        
        # Sauvegarder dans l'historique
        save_to_history(request.prompt, style, image_base64)
        
        print(f"Image générée avec succès en utilisant le style {style}")
        return {"image": image_base64}
    
    except Exception as e:
        print(f"Erreur lors de la génération: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération: {str(e)}")

@app.post("/transform", response_model=ImageResponse)
async def transform_image(
    image: UploadFile = File(...),
    style: str = Form(...),
    num_inference_steps: int = Form(30),
    guidance_scale: float = Form(7.5)
):
    try:
        # Sélection du modèle approprié en fonction du style
        selected_model = get_model_for_style(style)
        if selected_model is None:
            raise HTTPException(status_code=500, detail="Impossible de charger le modèle")
        
        # Lire et convertir l'image téléchargée
        contents = await image.read()
        input_image = Image.open(BytesIO(contents))
        
        # Redimensionner si nécessaire
        if input_image.width > 512 or input_image.height > 512:
            input_image.thumbnail((512, 512))
        
        # Préparation du prompt en fonction du style
        if style == "sksdisney":
            prompt = "Transform this image in Disney style"
        else:
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
        
        # Génération avec le modèle sélectionné
        result = selected_model(**params)
        
        # Conversion en base64
        image_base64 = pil_to_base64(result.images[0])
        
        # Sauvegarder dans l'historique
        save_to_history("Image transformée", style, image_base64)
        
        print(f"Image transformée avec succès en utilisant le style {style}")
        return {"image": image_base64}
    
    except Exception as e:
        print(f"Erreur lors de la transformation: {str(e)}")
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
    return {
        "status": "healthy", 
        "models_loaded": {
            "base": "base" in models,
            "disney": "disney" in models
        }
    }

# Point d'entrée pour le démarrage avec Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)