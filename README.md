# DreamBooth Disney Style API

Cette API permet de générer des images avec un style Disney en utilisant un modèle DreamBooth fine-tuné sur Stable Diffusion.

## Prérequis

- Docker et Docker Compose
- NVIDIA GPU avec les pilotes CUDA (pour l'inférence rapide)
- Docker NVIDIA Runtime (pour utiliser le GPU dans Docker)

## Structure du projet

```
ProjetIA/
├── app.py                   # Application FastAPI
├── configDreamBoothStyle.py # Script d'entraînement DreamBooth
├── Dockerfile               # Configuration Docker
├── docker-compose.yml       # Configuration Docker Compose
├── requirements.txt         # Dépendances Python
├── dreambooth_model_disney/ # Modèle fine-tuné
└── Datasets/                # Données d'entraînement
```

## Utilisation locale

1. Construire et démarrer le conteneur Docker :

```bash
docker-compose up --build
```

2. Accéder à l'API via :
   - Documentation Swagger : http://localhost:8000/docs
   - Documentation ReDoc : http://localhost:8000/redoc
   - Endpoint API : http://localhost:8000/generate

## Endpoints API

- `GET /` : Page d'accueil
- `GET /health` : Vérification de l'état de l'API
- `POST /generate` : Génération d'image

### Exemple de requête pour générer une image

```bash
curl -X 'POST' \
  'http://localhost:8000/generate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "a castle in the clouds, magical atmosphere",
  "num_inference_steps": 30,
  "guidance_scale": 7.5
}'
```

## Déploiement sur AWS

### Prérequis AWS

- Compte AWS
- AWS CLI configuré
- ECR (Elastic Container Registry)
- ECS (Elastic Container Service) ou EKS (Elastic Kubernetes Service)

### Étapes de déploiement

1. Créer un repository ECR
2. Authentifier Docker à ECR
3. Construire et pousser l'image Docker
4. Configurer un service ECS ou EKS
5. Déployer l'application

Voir le guide détaillé dans la section "Déploiement AWS" ci-dessous.

## Déploiement AWS

### 1. Créer un repository ECR

```bash
aws ecr create-repository --repository-name dreambooth-disney-api
```

### 2. Authentifier Docker à ECR

```bash
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
```

### 3. Construire et pousser l'image Docker

```bash
docker build -t dreambooth-disney-api .
docker tag dreambooth-disney-api:latest <account-id>.dkr.ecr.<region>.amazonaws.com/dreambooth-disney-api:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/dreambooth-disney-api:latest
```

### 4. Configurer un service ECS

Créez une définition de tâche ECS qui spécifie l'image Docker, la mémoire, le CPU, et les ports à exposer. Ensuite, créez un service ECS qui exécute cette tâche.

Pour les modèles d'IA nécessitant un GPU, utilisez des instances EC2 avec GPU (p2, p3, g4dn) et configurez ECS pour utiliser ces instances.

### 5. Configuration des variables d'environnement

Assurez-vous de configurer les variables d'environnement nécessaires dans votre service ECS/EKS :
- `MODEL_PATH` : Chemin vers le modèle dans le conteneur

### 6. Stockage du modèle

Pour les modèles volumineux, envisagez d'utiliser Amazon S3 pour stocker le modèle et de le télécharger au démarrage du conteneur, ou utilisez Amazon EFS pour monter un système de fichiers partagé.

## Bonnes pratiques pour la production

1. Sécurité :
   - Limitez les origines CORS
   - Ajoutez une authentification à l'API
   - Utilisez HTTPS

2. Performance :
   - Utilisez des instances avec GPU pour l'inférence
   - Configurez l'autoscaling en fonction de la charge

3. Monitoring :
   - Intégrez CloudWatch pour surveiller les métriques
   - Configurez des alertes pour les erreurs et la latence

4. Coûts :
   - Utilisez des instances Spot pour réduire les coûts
   - Configurez l'autoscaling pour réduire à zéro pendant les périodes d'inactivité
