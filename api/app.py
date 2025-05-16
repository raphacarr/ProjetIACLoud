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
styles_metadata = {}  # Métadonnées des styles chargées depuis S3

# Configuration S3
S3_BUCKET = "ia-cloud-models"
S3_DISNEY_MODEL_PATH = "disney"
LOCAL_MODELS_DIR = os.environ.get("MODEL_PATH", "../model")

# Fonction pour télécharger un modèle depuis S3
def download_model_from_s3(bucket_name, s3_path, local_path):
    try:
        log_with_timestamp("INFO", f"Téléchargement du modèle depuis S3: {bucket_name}/{s3_path}")
        
        # Vérification de l'espace disque disponible
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (1024 ** 3)
        log_with_timestamp("INFO", f"Espace disque disponible: {free_gb} GB")
        
        if free_gb < 2:  # Vérifier s'il y a au moins 2 GB d'espace libre
            log_with_timestamp("WARNING", "Espace disque insuffisant pour télécharger le modèle")
            return False
            
        # Créer un répertoire temporaire pour le téléchargement
        temp_dir = os.path.join(LOCAL_MODELS_DIR, f"temp_{uuid.uuid4().hex}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Téléchargement des fichiers
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        file_count = 0
        
        for obj in bucket.objects.filter(Prefix=s3_path):
            if obj.key == s3_path:  # Ignorer le dossier lui-même
                continue
                
            # Chemin relatif par rapport au préfixe
            rel_path = obj.key[len(s3_path):].lstrip('/')
            if not rel_path:  # Ignorer les entrées vides
                continue
                
            # Chemin local complet
            local_file_path = os.path.join(temp_dir, rel_path)
            
            # Créer les sous-répertoires si nécessaire
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            # Télécharger par morceaux pour économiser la mémoire
            log_with_timestamp("INFO", f"Téléchargement de {obj.key}")
            
            # Utiliser le client S3 au lieu de la ressource pour le téléchargement par morceaux
            s3_client = boto3.client('s3')
            
            try:
                # Télécharger par morceaux de 5 Mo
                with open(local_file_path, 'wb') as f:
                    s3_client.download_fileobj(
                        Bucket=bucket_name,
                        Key=obj.key,
                        Fileobj=f,
                        Config=boto3.s3.transfer.TransferConfig(
                            multipart_threshold=5 * 1024 * 1024,  # 5 Mo
                            max_concurrency=1,  # Un seul thread à la fois
                            multipart_chunksize=5 * 1024 * 1024,  # 5 Mo par morceau
                            use_threads=False  # Désactiver le multithreading
                        )
                    )
                    
                # Forcer la libération de la mémoire après chaque fichier
                import gc
                gc.collect()
                
                file_count += 1
                log_with_timestamp("INFO", f"Fichier {obj.key} téléchargé avec succès")
                
            except Exception as download_error:
                log_with_timestamp("ERROR", f"Erreur lors du téléchargement de {obj.key}: {str(download_error)}")
                # Continuer avec les autres fichiers
            
        log_with_timestamp("INFO", f"{file_count} fichiers téléchargés avec succès")
        
        # Si le téléchargement est réussi, déplacer les fichiers vers l'emplacement final
        if file_count > 0:
            # Supprimer l'ancien répertoire s'il existe
            if os.path.exists(local_path):
                shutil.rmtree(local_path)
            
            # Renommer le répertoire temporaire
            shutil.move(temp_dir, local_path)
            log_with_timestamp("INFO", f"Modèle installé avec succès dans {local_path}")
            return True
        else:
            log_with_timestamp("ERROR", "Aucun fichier trouvé à télécharger")
            shutil.rmtree(temp_dir)  # Nettoyer le répertoire temporaire
            return False
            
    except Exception as e:
        log_with_timestamp("ERROR", f"Erreur lors du téléchargement depuis S3: {str(e)}")
        log_with_timestamp("ERROR", traceback.format_exc())
        return False
# Fonction pour charger les métadonnées des styles
def load_styles_metadata():
    global styles_metadata
    try:
        # Chemin vers le fichier de métadonnées dans l'image Docker
        metadata_path = os.path.join(LOCAL_MODELS_DIR, "styles_metadata.json")
        
        # Charger depuis le fichier local
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                styles_metadata = json.load(f)
            log_with_timestamp("INFO", f"Métadonnées des styles chargées depuis le fichier local: {len(styles_metadata)} styles disponibles")
            return styles_metadata
            
        # Fallback sur S3 si le fichier local n'existe pas
        log_with_timestamp("WARNING", "Fichier de métadonnées non trouvé localement, tentative de chargement depuis S3")
        s3 = boto3.client('s3')
        response = s3.get_object(Bucket=S3_BUCKET, Key="styles_metadata.json")
        styles_metadata = json.loads(response['Body'].read().decode('utf-8'))
        
        # Sauvegarder localement pour utilisation future
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(styles_metadata, f)
            
        log_with_timestamp("INFO", f"Métadonnées des styles chargées depuis S3: {len(styles_metadata)} styles disponibles")
        return styles_metadata
    except Exception as e:
        log_with_timestamp("ERROR", f"Erreur lors du chargement des métadonnées de styles: {str(e)}")
        # Métadonnées par défaut si le chargement échoue
        styles_metadata = {
            "base_models": {
                "name": "Standard",
                "description": "Style standard de Stable Diffusion",
                "model_path": "base_models",
                "prompt_prefix": ""
            }
        }
        return styles_metadata

# Fonction pour charger un modèle
def load_model(style_id):
    global styles_metadata
    if not styles_metadata:
        load_styles_metadata()
        
    # Gestion du style par défaut
    if style_id not in styles_metadata:
        log_with_timestamp("WARNING", f"Style inconnu: {style_id}, utilisation du style de base")
        style_id = "base_models"  # Utiliser base_models comme style par défaut
    
    style_config = styles_metadata[style_id]
    model_path = os.path.join(LOCAL_MODELS_DIR, style_id)
    
    try:
        log_with_timestamp("INFO", f"=== DÉBUT CHARGEMENT MODÈLE {style_id.upper()} ===")
        
        # Vérifier si le modèle existe déjà dans l'image Docker
        if os.path.exists(model_path) and os.listdir(model_path):
            log_with_timestamp("INFO", f"Modèle {style_id} trouvé localement dans l'image Docker")
        else:
            log_with_timestamp("WARNING", f"Modèle {style_id} non trouvé dans l'image Docker")
            # Si vous voulez garder la possibilité de télécharger en fallback:
            log_with_timestamp("INFO", f"Tentative de téléchargement depuis S3...")
            if not download_model_from_s3(S3_BUCKET, style_config.get("s3_path", style_id), model_path):
                if style_id != "base_models":
                    log_with_timestamp("WARNING", f"Échec du téléchargement du modèle {style_id}, tentative avec le modèle de base")
                    return load_model("base_models")
                else:
                    raise Exception("Impossible de charger le modèle de base")
        
        # Chargement du modèle
        log_with_timestamp("INFO", f"Chargement du modèle {style_id} depuis {model_path}...")
        start_time = datetime.now()
        
        # Configurer le type de données selon le device
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Chargement du modèle
        model = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)
        
        # Optimisations CPU si nécessaire
        if device == "cpu":
            model.enable_attention_slicing()
            log_with_timestamp("INFO", "Attention slicing activé pour optimiser l'utilisation CPU")
        
        # Calcul du temps de chargement
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        log_with_timestamp("INFO", f"Modèle {style_id} chargé en {duration:.2f} secondes")
        log_with_timestamp("INFO", f"=== FIN CHARGEMENT MODÈLE {style_id.upper()} (SUCCÈS) ===")
        
        return model
        
    except Exception as e:
        log_with_timestamp("ERROR", f"Erreur lors du chargement du modèle {style_id}: {str(e)}")
        log_with_timestamp("ERROR", f"Traceback: {traceback.format_exc()}")
        log_with_timestamp("ERROR", f"=== FIN CHARGEMENT MODÈLE {style_id.upper()} (ÉCHEC) ===")
        
        if style_id != "base_models":
            log_with_timestamp("INFO", "Tentative de chargement du modèle de base à la place")
            return load_model("base_models")
        else:
            raise

# Fonction pour obtenir le modèle correspondant au style
def get_model_for_style(style):
    """Récupère ou charge le modèle correspondant au style demandé"""
    log_with_timestamp("INFO", f"Demande du modèle pour le style: {style}")
    global models, styles_metadata
    
    # S'assurer que les métadonnées sont chargées
    if not styles_metadata:
        load_styles_metadata()
    
    # Trouver l'ID du style correspondant
    style_id = "base"  # Par défaut
    for sid, sconfig in styles_metadata.items():
        if style == sid or style == sconfig.get("prompt_prefix", ""):
            style_id = sid
            break
    
    # Charger le modèle s'il n'est pas déjà en mémoire
    if style_id not in models:
        log_with_timestamp("INFO", f"Chargement du modèle pour le style {style_id}...")
        models[style_id] = load_model(style_id)
    
    return models[style_id]

# Fonction pour préparer le prompt en fonction du style
def prepare_prompt_for_style(prompt, style):
    """Prépare le prompt en ajoutant le préfixe du style si nécessaire"""
    global styles_metadata
    
    # S'assurer que les métadonnées sont chargées
    if not styles_metadata:
        load_styles_metadata()
    
    # Trouver l'ID du style correspondant
    style_id = "base"
    for sid, sconfig in styles_metadata.items():
        if style == sid or style == sconfig.get("prompt_prefix", ""):
            style_id = sid
            break
    
    # Récupérer le préfixe du style
    style_config = styles_metadata[style_id]
    prefix = style_config.get("prompt_prefix", "")
    
    # Ajouter le préfixe au prompt s'il n'est pas déjà présent
    if prefix and prefix not in prompt:
        return f"{prompt}, {prefix}"
    
    return prompt

# Fonction pour obtenir les paramètres par défaut du style
def get_default_params_for_style(style):
    """Récupère les paramètres par défaut pour un style donné"""
    global styles_metadata
    
    # S'assurer que les métadonnées sont chargées
    if not styles_metadata:
        load_styles_metadata()
    
    # Trouver l'ID du style correspondant
    style_id = "base"
    for sid, sconfig in styles_metadata.items():
        if style == sid or style == sconfig.get("prompt_prefix", ""):
            style_id = sid
            break
    
    # Récupérer les paramètres par défaut
    style_config = styles_metadata[style_id]
    return style_config.get("default_params", {})

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
    log_with_timestamp("INFO", "=== DÉMARRAGE DE L'API DREAM GENERATOR ===")
    log_with_timestamp("INFO", f"Version Python: {sys.version}")
    log_with_timestamp("INFO", f"Device: {device}")
    
    # Charger les métadonnées des styles au démarrage
    load_styles_metadata()
    
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
    
    # Précharger le modèle de base en arrière-plan
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
        style = request.style or "base"
        prompt = prepare_prompt_for_style(request.prompt, style)
        
        # Sélection du modèle approprié
        model = get_model_for_style(style)
        if model is None:
            raise HTTPException(status_code=500, detail="Impossible de charger le modèle")
        
        # Récupération des paramètres par défaut du style
        default_params = get_default_params_for_style(style)
        
        # Fusion des paramètres par défaut avec ceux de la requête
        num_inference_steps = request.num_inference_steps or default_params.get("num_inference_steps", 30)
        guidance_scale = request.guidance_scale or default_params.get("guidance_scale", 7.5)
        negative_prompt = request.negative_prompt or default_params.get("negative_prompt", "")
        
        # Génération de l'image
        generator = torch.Generator(device=device).manual_seed(42)  # Pour la reproductibilité
        
        # Préparation des paramètres
        params = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator
        }
        
        # Ajouter le negative_prompt s'il est défini
        if negative_prompt:
            params["negative_prompt"] = negative_prompt
            
        # Génération de l'image
        log_with_timestamp("INFO", f"Génération d'image avec prompt: '{prompt}'")
        start_time = datetime.now()
        result = model(**params)
        end_time = datetime.now()
        
        # Conversion de l'image en base64
        image = result.images[0]
        image_base64 = pil_to_base64(image)
        
        # Calcul du temps de génération
        duration = (end_time - start_time).total_seconds()
        log_with_timestamp("INFO", f"Image générée en {duration:.2f} secondes")
        
        # Sauvegarde dans l'historique
        entry = save_to_history(request.prompt, style, image_base64)
        
        return {"image": image_base64}
        
    except Exception as e:
        log_with_timestamp("ERROR", f"Erreur lors de la génération d'image: {str(e)}")
        log_with_timestamp("ERROR", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération d'image: {str(e)}")
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
        style_name = "standard"
        if style in styles_metadata:
            style_name = styles_metadata[style].get("name", style)
        
        prompt = f"Transform this image in {style_name} style"
        
        # Génération de l'image
        generator = torch.Generator(device=device).manual_seed(42)
        
        # Préparation des paramètres
        default_params = get_default_params_for_style(style)
        
        params = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps or default_params.get("num_inference_steps", 30),
            "guidance_scale": guidance_scale or default_params.get("guidance_scale", 7.5),
            "generator": generator
        }
        
        # Génération avec le modèle sélectionné
        log_with_timestamp("INFO", f"Transformation d'image avec style: '{style}'")
        start_time = datetime.now()
        result = selected_model(**params)
        end_time = datetime.now()
        
        # Calcul du temps de génération
        duration = (end_time - start_time).total_seconds()
        log_with_timestamp("INFO", f"Image transformée en {duration:.2f} secondes")
        
        # Conversion en base64
        image_base64 = pil_to_base64(result.images[0])
        
        # Sauvegarder dans l'historique
        save_to_history("Image transformée", style, image_base64)
        
        return {"image": image_base64}
    
    except Exception as e:
        log_with_timestamp("ERROR", f"Erreur lors de la transformation: {str(e)}")
        log_with_timestamp("ERROR", traceback.format_exc())
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
    global models, styles_metadata
    
    # S'assurer que les métadonnées sont chargées
    if not styles_metadata:
        load_styles_metadata()
    
    # Vérifier les modèles chargés en mémoire
    loaded_models = {style_id: style_id in models for style_id in styles_metadata.keys()}
    
    # Vérifier les modèles disponibles dans l'image Docker (préchargés)
    models_available = {}
    for style_id in styles_metadata.keys():
        model_path = os.path.join(LOCAL_MODELS_DIR, style_id)
        models_available[style_id] = os.path.exists(model_path) and len(os.listdir(model_path)) > 0
    
    # Récupérer les informations système
    system_info = {
        "device": device,
        "python_version": sys.version,
        "memory_available_gb": "N/A",
        "disk_space_available_gb": "N/A"
    }
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        system_info["memory_available_gb"] = f"{memory.available / (1024**3):.2f}"
        
        total, used, free = shutil.disk_usage("/")
        system_info["disk_space_available_gb"] = f"{free / (1024**3):.2f}"
    except Exception as e:
        system_info["error"] = str(e)
    
    return {
        "status": "healthy", 
        "models_loaded": loaded_models,          # Modèles chargés en mémoire
        "models_available": models_available,    # Modèles disponibles dans l'image Docker
        "styles_available": len(styles_metadata),
        "system_info": system_info
    }
# Point d'entrée pour le démarrage avec Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)