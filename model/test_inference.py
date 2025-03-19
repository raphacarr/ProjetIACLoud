import torch
from diffusers import StableDiffusionPipeline

def main():
    # Charger le modèle Stable Diffusion v1.5 depuis HuggingFace
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16
    )
    pipe.to("cuda")

    # Prompt simple de test
    prompt = "a puppy, high resolution photography, ultra realistic, 4k"

    # Génération d'une image
    image = pipe(prompt, num_inference_steps=25).images[0]
    image.save("test_puppy.png")

    print("Image générée : test_puppy.png")

if __name__ == "__main__":
    main()
