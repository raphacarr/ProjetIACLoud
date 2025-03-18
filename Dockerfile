FROM python:3.10-slim

WORKDIR /app

# Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers de dépendances et installation
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code de l'application
COPY app.py .

# Copie du modèle fine-tuné (si disponible)
COPY dreambooth_model_disney/ /app/dreambooth_model_disney/

# Variable d'environnement pour le chemin du modèle
ENV MODEL_PATH=/app/dreambooth_model_disney

# Exposition du port
EXPOSE 8000

# Commande de démarrage
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
