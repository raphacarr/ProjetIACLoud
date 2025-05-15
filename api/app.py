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
import logging
import sys
import traceback
from datetime import datetime

#config du système de log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

#Créer un logger global
logger = logging.getLogger("api")
logger.setLevel(logging.INFO)

#Désactiver la mise en mémoire tampon pour les sorties standard
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Utilitaire pour ajouter des timestamps aux logs
def log_with_timestamp(level, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    if level == "INFO":
        logger.info(f"[{timestamp}] {message}")
    elif level == "ERROR":
        logger.error(f"[{timestamp}] {message}")
    elif level == "WARNING":
        logger.warning(f"[{timestamp}] {message}")
    elif level == "DEBUG":
        logger.debug(f"[{timestamp}] {message}")


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
    color: str
    available: bool

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
        print(f"Début du téléchargement depuis S3: {bucket_name}/{s3_path}")
        # Liste les fichiers dans le bucket pour vérifier
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        print(f"Contenu du bucket dans le chemin {s3_path}:")
        for obj in bucket.objects.filter(Prefix=s3_path):
            print(f"  - {obj.key}")
        
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
    log_with_timestamp("INFO", "=== DÉBUT CHARGEMENT MODÈLE BASE ===")
    
    try:
        # Vérifier le chemin du modèle
        model_path = f"{LOCAL_MODELS_DIR}/base"
        log_with_timestamp("INFO", f"Chemin du modèle: {model_path}")
        
        # Vérifier si le dossier existe
        if os.path.exists(model_path):
            files = os.listdir(model_path)
            log_with_timestamp("INFO", f"Dossier du modèle trouvé, contient {len(files)} fichiers")
            if len(files) > 0:
                log_with_timestamp("INFO", f"Exemples de fichiers: {', '.join(files[:5])}")
        else:
            log_with_timestamp("WARNING", f"Le dossier du modèle n'existe pas: {model_path}")
            log_with_timestamp("INFO", "Tentative de téléchargement depuis S3...")
            
            # Ici, ajoutez votre code de téléchargement depuis S3 si nécessaire
        
        # Chargement du modèle
        log_with_timestamp("INFO", "Chargement du modèle avec StableDiffusionPipeline...")
        start_time = datetime.now()
        
        # Configurer le type de données selon le device
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        log_with_timestamp("INFO", f"Utilisation du type de données: {torch_dtype}")
        
        model = StableDiffusionPipeline.from_pretrained(
            model_path, 
            torch_dtype=torch_dtype
        )
        
        # Transfert vers le device
        log_with_timestamp("INFO", f"Transfert du modèle vers {device}...")
        model = model.to(device)
        
        # Calcul du temps de chargement
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        log_with_timestamp("INFO", f"Modèle chargé en {duration:.2f} secondes")
        
        log_with_timestamp("INFO", "=== FIN CHARGEMENT MODÈLE BASE (SUCCÈS) ===")
        return model
        
    except Exception as e:
        log_with_timestamp("ERROR", f"Erreur lors du chargement du modèle: {str(e)}")
        log_with_timestamp("ERROR", f"Traceback: {traceback.format_exc()}")
        log_with_timestamp("ERROR", "=== FIN CHARGEMENT MODÈLE BASE (ÉCHEC) ===")
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
    log_with_timestamp("INFO", f"Demande du modèle pour le style: {style}")
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

async def startup_event():
    log_with_timestamp("INFO", "=== DÉMARRAGE DE L'API DREAM GENERATOR ===")
    log_with_timestamp("INFO", f"Version Python: {sys.version}")
    log_with_timestamp("INFO", f"Device: {device}")
    
    # Vérification de la mémoire
    try:
        import psutil
        memory = psutil.virtual_memory()
        log_with_timestamp("INFO", f"Mémoire totale: {memory.total / (1024**3):.2f} GB")
        log_with_timestamp("INFO", f"Mémoire disponible: {memory.available / (1024**3):.2f} GB")
    except Exception as e:
        log_with_timestamp("WARNING", f"Impossible de vérifier la mémoire: {str(e)}")
    
    # Vérification de CUDA
    if device == "cuda":
        log_with_timestamp("INFO", f"CUDA disponible: {torch.cuda.get_device_name(0)}")
        log_with_timestamp("INFO", f"VRAM totale: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        log_with_timestamp("INFO", "CUDA non disponible, utilisation du CPU")
    
    # Vérification de l'accès à S3
    try:
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(Bucket=S3_BUCKET, MaxKeys=1)
        log_with_timestamp("INFO", f"Connexion S3 réussie au bucket: {S3_BUCKET}")
    except Exception as e:
        log_with_timestamp("ERROR", f"Erreur de connexion S3: {str(e)}")
    
    # Chargement du modèle de base en arrière-plan
    import threading
    threading.Thread(target=lambda: get_model_for_style("base")).start()
    log_with_timestamp("INFO", "Chargement du modèle de base en arrière-plan")
    
    # Créer le répertoire des modèles s'il n'existe pas
    os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)
    log_with_timestamp("INFO", f"Répertoire des modèles: {LOCAL_MODELS_DIR}")

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
    try:
        # Se connecter à S3
        s3 = boto3.client('s3')
        
        # Télécharger le fichier de métadonnées des styles
        try:
            response = s3.get_object(Bucket=S3_BUCKET, Key="styles_metadata.json")
            styles_metadata = json.loads(response['Body'].read().decode('utf-8'))
        except Exception as e:
            print(f"Erreur lors du chargement des métadonnées: {str(e)}")
            # Utiliser des métadonnées par défaut si le fichier n'existe pas
            styles_metadata = {
                "base": {"name": "Standard", "description": "Style standard de Stable Diffusion"},
                "disney": {"name": "Disney", "description": "Style inspiré des films Disney"}
            }
        
        # Vérifier quels modèles sont disponibles dans S3
        available_models = set()
        for obj in s3.list_objects_v2(Bucket=S3_BUCKET, Delimiter='/').get('CommonPrefixes', []):
            model_name = obj.get('Prefix', '').strip('/')
            if model_name:
                available_models.add(model_name)
        
        # Liste de couleurs vives pour une génération pseudo-aléatoire mais déterministe
        colors = [
            "#FF5C5C", "#FF9A5C", "#FFD15C", "#9AFF5C", "#5CFF9A", 
            "#5CFFD1", "#5CD1FF", "#5C9AFF", "#9A5CFF", "#D15CFF", 
            "#FF5CD1", "#FF5C9A", "#5CFFFF", "#5C5CFF", "#FF5C5C"
        ]
        
        # Créer la liste des styles avec leur disponibilité
        styles = []
        for i, (model_id, metadata) in enumerate(styles_metadata.items()):
            # Sélection déterministe d'une couleur basée sur l'index
            color_index = hash(model_id) % len(colors)
            color = colors[color_index]
            
            styles.append({
                "id": model_id,
                "name": metadata.get("name", model_id),
                "description": metadata.get("description", ""),
                "color": color,
                "available": model_id in available_models
            })
        
        return styles
    except Exception as e:
        print(f"Erreur lors de la vérification des modèles S3: {str(e)}")
        # Retourner une liste de secours en cas d'erreur
        return [
            {"id": "base", "name": "Standard", "description": "Style standard", "color": "#6a11cb", "available": True},
            {"id": "disney", "name": "Disney", "description": "Style Disney", "color": "#ff9d00", "available": True}
        ]

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