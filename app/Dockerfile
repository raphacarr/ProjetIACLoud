# Utiliser l'image Node.js pour le développement
FROM node:20-alpine

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances
COPY package.json package-lock.json ./

# Installer les dépendances
RUN npm install

# Copier le reste des fichiers de l'application
COPY . .

# Exposer le port 3000 (port par défaut de React en développement)
EXPOSE 3000

# Démarrer l'application en mode développement
CMD ["npm", "start"]
