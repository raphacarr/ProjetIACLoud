import axios from 'axios';

// Créer une instance axios avec une configuration de base
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // Augmenter le timeout pour les opérations de génération d'images
});

// Service pour générer une image à partir d'un prompt
export const generateImage = async (prompt, style = null) => {
  try {
    // Préparer les données de la requête
    const requestData = {
      prompt,
      style,
      num_inference_steps: 30,
      guidance_scale: 7.5
    };
    
    const response = await api.post('/generate', requestData);
    
    // L'API renvoie l'image en base64, on la convertit en URL
    if (response.data && response.data.image) {
      return `data:image/png;base64,${response.data.image}`;
    } else {
      throw new Error('Format de réponse invalide');
    }
  } catch (error) {
    console.error('Error generating image:', error);
    throw error;
  }
};

// Service pour transformer une image existante avec un style
export const transformImage = async (image, style) => {
  try {
    // Créer un FormData pour envoyer l'image
    const formData = new FormData();
    formData.append('image', image);
    formData.append('style', style);
    formData.append('num_inference_steps', 30);
    formData.append('guidance_scale', 7.5);
    
    const response = await api.post('/transform', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    if (response.data && response.data.image) {
      return `data:image/png;base64,${response.data.image}`;
    } else {
      throw new Error('Format de réponse invalide');
    }
  } catch (error) {
    console.error('Error transforming image:', error);
    throw error;
  }
};

// Service pour récupérer les styles disponibles
export const getAvailableStyles = async () => {
  try {
    const response = await api.get('/styles');
    
    if (response.data && Array.isArray(response.data)) {
      return response.data;
    } else {
      // Si l'API n'est pas disponible, retourner des styles par défaut
      return [
        { id: "sksdisney", name: "Disney", description: "Style inspiré des films Disney" },
        { id: "anime", name: "Anime", description: "Style manga japonais" },
        { id: "pixar", name: "Pixar", description: "Style des films Pixar" },
        { id: "watercolor", name: "Aquarelle", description: "Style peinture à l'aquarelle" },
        { id: "comic", name: "Comic", description: "Style bande dessinée" }
      ];
    }
  } catch (error) {
    console.error('Error fetching styles:', error);
    // Retourner des styles par défaut en cas d'erreur
    return [
      { id: "sksdisney", name: "Disney", description: "Style inspiré des films Disney" },
      { id: "anime", name: "Anime", description: "Style manga japonais" },
      { id: "pixar", name: "Pixar", description: "Style des films Pixar" }
    ];
  }
};

// Service pour récupérer l'historique des images générées
export const getImageHistory = async () => {
  try {
    const response = await api.get('/history');
    
    if (response.data && Array.isArray(response.data)) {
      return response.data;
    } else {
      return [];
    }
  } catch (error) {
    console.error('Error fetching image history:', error);
    return [];
  }
};

// Vérifier si l'API est disponible
export const checkApiHealth = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    console.error('API health check failed:', error);
    return { status: 'unhealthy' };
  }
};

export default {
  generateImage,
  transformImage,
  getAvailableStyles,
  getImageHistory,
  checkApiHealth
};
