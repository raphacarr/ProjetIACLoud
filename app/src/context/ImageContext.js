import React, { createContext, useState, useContext, useEffect } from 'react';
import { toast } from 'react-toastify';
import { generateImage, transformImage, getAvailableStyles, getImageHistory } from '../services/api';

// Créer le contexte
const ImageContext = createContext();

// Hook personnalisé pour utiliser le contexte
export const useImageContext = () => useContext(ImageContext);

// Fournisseur du contexte
export const ImageProvider = ({ children }) => {
  // États
  const [generatedImage, setGeneratedImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [availableStyles, setAvailableStyles] = useState([]);
  const [imageHistory, setImageHistory] = useState([]);
  
  // Récupérer les styles disponibles au chargement
  useEffect(() => {
  const fetchStyles = async () => {
    try {
      const styles = await getAvailableStyles();
      
      // Séparation des styles disponibles et en développement
      const availableStyles = styles.filter(style => style.available);
      const inDevelopmentStyles = styles.filter(style => !style.available);
      
      // Combiner les deux listes, avec les styles disponibles en premier
      setAvailableStyles([
        ...availableStyles,
        ...inDevelopmentStyles.map(style => ({
          ...style,
          inDevelopment: true
        }))
      ]);
    } catch (error) {
      toast.error('Erreur lors du chargement des styles');
      // Utiliser des styles par défaut en cas d'échec
      setAvailableStyles([
        { id: 'base', name: 'Standard', description: 'Style standard de Stable Diffusion', available: true },
        { id: 'disney', name: 'Disney', description: 'Style inspiré des films Disney', available: true }
      ]);
    }
  };
  
  fetchStyles();
}, []);
  
  // Récupérer l'historique des images au chargement
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        // En attendant l'implémentation de l'API, utilisons un historique fictif
        const mockHistory = [
          { id: 1, prompt: 'Un château dans les nuages', style: 'disney', imageUrl: 'https://via.placeholder.com/300', createdAt: new Date().toISOString() },
          { id: 2, prompt: 'Un dragon majestueux', style: 'anime', imageUrl: 'https://via.placeholder.com/300', createdAt: new Date().toISOString() }
        ];
        
        setImageHistory(mockHistory);
        
        // Décommenter quand l'API sera prête
        // const history = await getImageHistory();
        // setImageHistory(history);
      } catch (error) {
        toast.error('Erreur lors du chargement de l\'historique');
      }
    };
    
    fetchHistory();
  }, []);
  
  // Fonction pour générer une image
  const handleGenerateImage = async (prompt, style = null) => {
    setIsLoading(true);
    setGeneratedImage(null);
    
    try {
      const imageUrl = await generateImage(prompt, style);
      setGeneratedImage(imageUrl);
      
      // Ajouter à l'historique local
      const newImage = {
        id: Date.now(),
        prompt,
        style,
        imageUrl,
        createdAt: new Date().toISOString()
      };
      
      setImageHistory(prev => [newImage, ...prev]);
      toast.success('Image générée avec succès !');
    } catch (error) {
      toast.error('Erreur lors de la génération de l\'image');
    } finally {
      setIsLoading(false);
    }
  };
  
  // Fonction pour transformer une image
  const handleTransformImage = async (image, style) => {
    setIsLoading(true);
    setGeneratedImage(null);
    
    try {
      const imageUrl = await transformImage(image, style);
      setGeneratedImage(imageUrl);
      
      // Ajouter à l'historique local
      const newImage = {
        id: Date.now(),
        prompt: 'Image transformée',
        style,
        imageUrl,
        createdAt: new Date().toISOString()
      };
      
      setImageHistory(prev => [newImage, ...prev]);
      toast.success('Image transformée avec succès !');
    } catch (error) {
      toast.error('Erreur lors de la transformation de l\'image');
    } finally {
      setIsLoading(false);
    }
  };
  
  // Valeur du contexte
  const value = {
    generatedImage,
    isLoading,
    availableStyles,
    imageHistory,
    generateImage: handleGenerateImage,
    transformImage: handleTransformImage
  };
  
  return (
    <ImageContext.Provider value={value}>
      {children}
    </ImageContext.Provider>
  );
};

export default ImageContext;
