#!/bin/bash

# Variables à configurer
AWS_REGION="eu-west-3"  # Région AWS (à modifier selon votre région)
ECR_REPO_NAME="dreambooth-disney-api"  # Nom du repository ECR
IMAGE_TAG="latest"  # Tag de l'image Docker

# Récupération de l'ID de compte AWS
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
if [ $? -ne 0 ]; then
    echo "Erreur: Impossible de récupérer l'ID de compte AWS. Vérifiez votre configuration AWS CLI."
    exit 1
fi

# Construction de l'URI ECR
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_REPO_URI="${ECR_URI}/${ECR_REPO_NAME}"

echo "=== Déploiement de l'API DreamBooth Disney Style sur AWS ==="
echo "Région AWS: ${AWS_REGION}"
echo "Repository ECR: ${ECR_REPO_NAME}"
echo "URI ECR: ${ECR_REPO_URI}"

# 1. Création du repository ECR (s'il n'existe pas déjà)
echo "1. Création du repository ECR..."
aws ecr describe-repositories --repository-names ${ECR_REPO_NAME} > /dev/null 2>&1
if [ $? -ne 0 ]; then
    aws ecr create-repository --repository-name ${ECR_REPO_NAME}
    echo "Repository ECR créé: ${ECR_REPO_NAME}"
else
    echo "Repository ECR existe déjà: ${ECR_REPO_NAME}"
fi

# 2. Authentification à ECR
echo "2. Authentification à ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_URI}
if [ $? -ne 0 ]; then
    echo "Erreur: Échec de l'authentification à ECR."
    exit 1
fi
echo "Authentification réussie."

# 3. Construction de l'image Docker
echo "3. Construction de l'image Docker..."
docker build -t ${ECR_REPO_NAME}:${IMAGE_TAG} .
if [ $? -ne 0 ]; then
    echo "Erreur: Échec de la construction de l'image Docker."
    exit 1
fi
echo "Image Docker construite avec succès."

# 4. Tag de l'image pour ECR
echo "4. Tag de l'image pour ECR..."
docker tag ${ECR_REPO_NAME}:${IMAGE_TAG} ${ECR_REPO_URI}:${IMAGE_TAG}
if [ $? -ne 0 ]; then
    echo "Erreur: Échec du tag de l'image."
    exit 1
fi
echo "Image taguée avec succès."

# 5. Push de l'image vers ECR
echo "5. Push de l'image vers ECR..."
docker push ${ECR_REPO_URI}:${IMAGE_TAG}
if [ $? -ne 0 ]; then
    echo "Erreur: Échec du push de l'image vers ECR."
    exit 1
fi
echo "Image poussée avec succès vers ECR: ${ECR_REPO_URI}:${IMAGE_TAG}"

echo "=== Déploiement terminé avec succès ==="
echo "L'image Docker est maintenant disponible dans ECR."
echo "Pour déployer sur ECS/EKS, utilisez l'URI d'image suivant:"
echo "${ECR_REPO_URI}:${IMAGE_TAG}"
echo ""
echo "Étapes suivantes recommandées:"
echo "1. Créer une définition de tâche ECS (avec l'URI d'image ci-dessus)"
echo "2. Créer un service ECS basé sur cette définition de tâche"
echo "3. Configurer un Application Load Balancer pour exposer l'API"
