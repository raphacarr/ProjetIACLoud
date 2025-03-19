# DreamStyle Generator

Application web de génération et transformation d'images utilisant Stable Diffusion et React.

## Architecture

Le projet est composé de deux parties principales :

1. **Frontend (React)** : Interface utilisateur permettant de générer des images à partir de prompts et de transformer des images existantes.
2. **Backend (FastAPI)** : API qui utilise Stable Diffusion pour générer et transformer des images.

## Déploiement sur AWS App Runner

### Prérequis

- Un compte AWS avec accès à App Runner et ECR
- AWS CLI configuré avec les bonnes permissions
- Docker installé localement

### Étapes de déploiement

#### 1. Création des dépôts ECR

Créez deux dépôts ECR pour stocker les images Docker :

```bash
aws ecr create-repository --repository-name react-app-ia-cloud-projet
aws ecr create-repository --repository-name fastapi-api-ia-cloud-projet
```

#### 2. Configuration des variables d'environnement

Pour le frontend (app/.env) :
```
REACT_APP_API_URL=<URL_DE_VOTRE_API_DEPLOYEE>
```

Pour le backend (api/.env) :
```
MODEL_PATH=/app/model
```

#### 3. Construction et déploiement via AWS CodeBuild

1. Créez deux projets CodeBuild, un pour le frontend et un pour le backend.
2. Utilisez les fichiers buildspec.app.yml et buildspec.api.yml comme configurations.
3. Configurez les variables d'environnement nécessaires dans CodeBuild :
   - ACCOUNT_ID : Votre ID de compte AWS
   - AWS_DEFAULT_REGION : La région AWS que vous utilisez

#### 4. Configuration d'App Runner

1. Dans la console AWS, accédez à App Runner.
2. Créez un nouveau service en sélectionnant l'image ECR pour le frontend.
3. Configurez les paramètres du service (mémoire, CPU, etc.).
4. Répétez le processus pour l'API backend.

#### 5. Configuration du CORS

Assurez-vous que l'API backend autorise les requêtes CORS depuis l'URL de votre frontend déployé.

## Développement local

### Frontend

```bash
cd app
npm install
npm start
```

### Backend

```bash
cd api
pip install -r requirements.txt
uvicorn app:app --reload
```

## Fonctionnalités

- Génération d'images à partir de prompts textuels
- Transformation d'images existantes avec différents styles
- Historique des images générées
- Sélection de styles prédéfinis

## Technologies utilisées

- **Frontend** : React, styled-components, axios, react-router-dom
- **Backend** : FastAPI, Stable Diffusion, PyTorch
- **Déploiement** : AWS App Runner, ECR, CodeBuild
