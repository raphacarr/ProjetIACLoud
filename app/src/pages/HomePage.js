import React, { useState } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { FaMagic, FaSpinner } from 'react-icons/fa';
import { toast } from 'react-toastify';
import { useImageContext } from '../context/ImageContext';
import StyleSelector from '../components/StyleSelector';

const HomeContainer = styled.div`
  max-width: 1000px;
  margin: 2rem auto;
  padding: 0 1.5rem;
`;

const Hero = styled.div`
  text-align: center;
  margin-bottom: 3rem;
`;

const Title = styled(motion.h1)`
  font-size: 2.5rem;
  background: linear-gradient(to right, var(--primary-color), var(--secondary-color));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 1rem;
`;

const Subtitle = styled.p`
  font-size: 1.2rem;
  color: var(--text-secondary);
  max-width: 700px;
  margin: 0 auto 2rem;
`;

const InputContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1rem;
  margin-bottom: 2rem;
`;

const PromptInput = styled.input`
  padding: 1rem;
  font-size: 1.1rem;
  border-radius: var(--border-radius);
  border: 1px solid #ddd;
  
  &:focus {
    border-color: var(--primary-color);
    box-shadow: 0 0 0 2px rgba(106, 17, 203, 0.2);
  }
`;

const GenerateButton = styled(motion.button)`
  padding: 1rem 2rem;
  font-size: 1.1rem;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  background: linear-gradient(to right, var(--primary-color), var(--secondary-color));
  color: white;
  border: none;
  border-radius: var(--border-radius);
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
  }
  
  &:disabled {
    opacity: 0.7;
    cursor: not-allowed;
    transform: none;
  }
`;

const ResultContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-top: 2rem;
`;

const ImageContainer = styled(motion.div)`
  max-width: 512px;
  width: 100%;
  border-radius: var(--border-radius);
  overflow: hidden;
  box-shadow: var(--box-shadow);
  background: white;
`;

const GeneratedImage = styled.img`
  width: 100%;
  height: auto;
  display: block;
`;

const LoadingSpinner = styled(motion.div)`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  margin: 2rem 0;
  
  svg {
    font-size: 3rem;
    color: var(--primary-color);
  }
`;

const HomePage = () => {
  const [prompt, setPrompt] = useState('');
  const [selectedStyle, setSelectedStyle] = useState(null);
  const { generateImage, isLoading, generatedImage } = useImageContext();
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!prompt.trim()) {
      toast.error('Veuillez entrer un prompt');
      return;
    }
    
    try {
      await generateImage(prompt, selectedStyle);
    } catch (error) {
      toast.error('Erreur lors de la génération de l\'image');
    }
  };
  
  return (
    <HomeContainer>
      <Hero>
        <Title
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          DreamStyle Generator
        </Title>
        <Subtitle>
          Transformez vos idées en magnifiques images avec notre générateur d'images IA.
          Choisissez un style et décrivez ce que vous souhaitez voir.
        </Subtitle>
      </Hero>
      
      <form onSubmit={handleSubmit}>
        <InputContainer>
          <PromptInput
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Décrivez l'image que vous souhaitez générer..."
            disabled={isLoading}
          />
          
          <StyleSelector
            selectedStyle={selectedStyle}
            onSelectStyle={setSelectedStyle}
          />
          
          <GenerateButton
            type="submit"
            disabled={isLoading || !prompt.trim()}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            {isLoading ? (
              <>
                <FaSpinner className="spin" /> Génération en cours...
              </>
            ) : (
              <>
                <FaMagic /> Générer l'image
              </>
            )}
          </GenerateButton>
        </InputContainer>
      </form>
      
      {isLoading && (
        <LoadingSpinner
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <FaSpinner className="spin" />
          <p>Génération de votre image en cours...</p>
          <p>Cela peut prendre quelques instants</p>
        </LoadingSpinner>
      )}
      
      {generatedImage && !isLoading && (
        <ResultContainer>
          <ImageContainer
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <GeneratedImage src={generatedImage} alt={prompt} />
          </ImageContainer>
        </ResultContainer>
      )}
    </HomeContainer>
  );
};

export default HomePage;
