# ---- Étape 1 : Construire l'app React ----
FROM node:20-alpine AS builder

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de dépendances depuis le dossier "app"
COPY app/package*.json ./

# Installer les dépendances
RUN npm install

# Copier l'ensemble du code source depuis "app"
COPY app/ .

# Copier le fichier .env.production pour le build de production
COPY app/.env.production ./.env.production

# Construire l'application (génère le dossier "build")
RUN npm run build

# ---- Étape 2 : Servir l'app avec Nginx ----
FROM nginx:alpine

# Copier les fichiers buildés dans le répertoire de Nginx
COPY --from=builder /app/build /usr/share/nginx/html

# Copier une configuration Nginx personnalisée pour gérer les routes React
COPY app/nginx.conf /etc/nginx/conf.d/default.conf

# Exposer le port 80 (où Nginx écoute par défaut)
EXPOSE 80

# Démarrer Nginx
CMD ["nginx", "-g", "daemon off;"]
