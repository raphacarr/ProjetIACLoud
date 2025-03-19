# test_DreamBooth.py

import torch
from diffusers import StableDiffusionPipeline

def main():
    # Chemin vers ton modèle fine-tuné DreamBooth
    model_path = "E:/3-Travail/3 - ProjetIA/dreambooth_model_disney"

    # Charger le pipeline avec les poids fine-tunés
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    )
    pipe.to("cuda")

    # Prompt pour tester le style "Disney"
    prompt = "a castle in the clouds, sksdisneystyle, magical atmosphere"

    # Génération
    image = pipe(prompt, num_inference_steps=30).images[0]
    image.save("castle_disney.png")

    print("Image générée : castle_disney.png")

if __name__ == "__main__":
    main()
