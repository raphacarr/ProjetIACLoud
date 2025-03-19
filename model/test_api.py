import requests
import json
import base64
from PIL import Image
from io import BytesIO
import argparse

def generate_image(prompt, num_steps=30, guidance_scale=7.5, negative_prompt=None, api_url="http://localhost:8000"):
    """
    Génère une image via l'API DreamBooth Disney Style
    
    Args:
        prompt (str): Description de l'image à générer
        num_steps (int): Nombre d'étapes d'inférence
        guidance_scale (float): Échelle de guidance
        negative_prompt (str): Prompt négatif (optionnel)
        api_url (str): URL de l'API
    
    Returns:
        PIL.Image: Image générée
    """
    # Préparation de la requête
    endpoint = f"{api_url}/generate"
    payload = {
        "prompt": prompt,
        "num_inference_steps": num_steps,
        "guidance_scale": guidance_scale
    }
    
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    
    # Envoi de la requête
    print(f"Envoi de la requête à {endpoint}...")
    print(f"Prompt: {prompt}")
    
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        
        # Traitement de la réponse
        result = response.json()
        image_data = base64.b64decode(result["image"])
        image = Image.open(BytesIO(image_data))
        
        # Sauvegarde de l'image
        output_file = "generated_image.png"
        image.save(output_file)
        print(f"Image générée et sauvegardée sous: {output_file}")
        
        return image
    
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la requête: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Détails de l'erreur: {e.response.text}")
        return None

def check_health(api_url="http://localhost:8000"):
    """Vérifie l'état de l'API"""
    try:
        response = requests.get(f"{api_url}/health")
        response.raise_for_status()
        print(f"Statut de l'API: {response.json()}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la vérification de l'état de l'API: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test de l'API DreamBooth Disney Style")
    parser.add_argument("--prompt", type=str, default="a castle in the clouds, magical atmosphere",
                        help="Description de l'image à générer")
    parser.add_argument("--steps", type=int, default=30,
                        help="Nombre d'étapes d'inférence")
    parser.add_argument("--guidance", type=float, default=7.5,
                        help="Échelle de guidance")
    parser.add_argument("--negative", type=str, default=None,
                        help="Prompt négatif")
    parser.add_argument("--url", type=str, default="http://localhost:8000",
                        help="URL de l'API")
    
    args = parser.parse_args()
    
    # Vérification de l'état de l'API
    if check_health(args.url):
        # Génération de l'image
        generate_image(
            prompt=args.prompt,
            num_steps=args.steps,
            guidance_scale=args.guidance,
            negative_prompt=args.negative,
            api_url=args.url
        )
