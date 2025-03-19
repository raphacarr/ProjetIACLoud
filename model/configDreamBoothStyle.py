import subprocess

def run_dreambooth_disney():
    """
    Lance l'entraînement DreamBooth pour un style 'Disney'
    """
    train_script = r"D:\Cours\Master2\IA_Cloud\ProjetIA\diffusers\examples\dreambooth\train_dreambooth.py"
    command = [
        "accelerate", "launch", train_script,
        "--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5",
        "--instance_data_dir=D:\Cours\Master2\IA_Cloud\ProjetIA\Datasets\disneyStyle",
        "--output_dir=D:\Cours\Master2\IA_Cloud\ProjetIA\dreambooth_model_disney",
        "--instance_prompt=sksdisneystyle",
        "--resolution=512",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=1",
        "--learning_rate=5e-6",
        "--max_train_steps=800",
        "--mixed_precision=fp16"
    ]
    subprocess.run(command, check=True)

def main():
    print("Lancement de l'entraînement DreamBooth pour le style Disney...")
    run_dreambooth_disney()
    print("Entraînement terminé.")

if __name__ == "__main__":
    main()
