import React, { useState, useRef } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { FaUpload, FaSpinner, FaMagic } from 'react-icons/fa';
import { toast } from 'react-toastify';
import { useImageContext } from '../context/ImageContext';
import StyleSelector from '../components/StyleSelector';

const PageContainer = styled.div`
  max-width: 1000px;
  margin: 2rem auto;
  padding: 0 1.5rem;
`;

const PageTitle = styled(motion.h1)`
  font-size: 2.5rem;
  background: linear-gradient(to right, var(--primary-color), var(--secondary-color));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 1rem;
  text-align: center;
`;

const PageDescription = styled.p`
  font-size: 1.2rem;
  color: var(--text-secondary);
  max-width: 700px;
  margin: 0 auto 2rem;
  text-align: center;
`;

const TransformContainer = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
  margin-top: 2rem;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const UploadSection = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
`;

const ResultSection = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
`;

const UploadBox = styled.div`
  border: 2px dashed ${props => props.isDragging ? 'var(--primary-color)' : '#ddd'};
  border-radius: var(--border-radius);
  padding: 2rem;
  text-align: center;
  background: ${props => props.isDragging ? 'rgba(106, 17, 203, 0.05)' : 'white'};
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    border-color: var(--primary-color);
    background: rgba(106, 17, 203, 0.05);
  }
`;

const UploadIcon = styled(FaUpload)`
  font-size: 3rem;
  color: var(--primary-color);
  margin-bottom: 1rem;
`;

const UploadText = styled.p`
  margin-bottom: 0.5rem;
  font-size: 1.1rem;
`;

const UploadSubtext = styled.p`
  color: var(--text-light);
  font-size: 0.9rem;
`;

const HiddenInput = styled.input`
  display: none;
`;

const PreviewContainer = styled.div`
  position: relative;
  border-radius: var(--border-radius);
  overflow: hidden;
  box-shadow: var(--box-shadow);
  background: white;
  
  img {
    width: 100%;
    height: auto;
    display: block;
  }
`;

const RemoveButton = styled.button`
  position: absolute;
  top: 10px;
  right: 10px;
  background: rgba(0, 0, 0, 0.6);
  color: white;
  border: none;
  border-radius: 50%;
  width: 30px;
  height: 30px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  
  &:hover {
    background: rgba(0, 0, 0, 0.8);
  }
`;

const TransformButton = styled(motion.button)`
  padding: 1rem;
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

const TransformPage = () => {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedStyle, setSelectedStyle] = useState(null);
  const fileInputRef = useRef(null);
  
  const { transformImage, isLoading, generatedImage } = useImageContext();
  
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      processFile(file);
    }
  };
  
  const processFile = (file) => {
    if (!file.type.match('image.*')) {
      toast.error('Veuillez sélectionner une image');
      return;
    }
    
    if (file.size > 5 * 1024 * 1024) {
      toast.error('L\'image ne doit pas dépasser 5 Mo');
      return;
    }
    
    setUploadedImage(file);
    const reader = new FileReader();
    reader.onload = () => {
      setPreviewUrl(reader.result);
    };
    reader.readAsDataURL(file);
  };
  
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };
  
  const handleDragLeave = () => {
    setIsDragging(false);
  };
  
  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };
  
  const handleUploadClick = () => {
    fileInputRef.current.click();
  };
  
  const handleRemoveImage = () => {
    setUploadedImage(null);
    setPreviewUrl(null);
  };
  
  const handleTransform = async () => {
    if (!uploadedImage) {
      toast.error('Veuillez d\'abord télécharger une image');
      return;
    }
    
    if (!selectedStyle) {
      toast.error('Veuillez sélectionner un style');
      return;
    }
    
    try {
      await transformImage(uploadedImage, selectedStyle);
    } catch (error) {
      toast.error('Erreur lors de la transformation de l\'image');
    }
  };
  
  return (
    <PageContainer>
      <PageTitle
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        Transformer une Image
      </PageTitle>
      <PageDescription>
        Téléchargez une image et transformez-la avec le style de votre choix.
      </PageDescription>
      
      <StyleSelector
        selectedStyle={selectedStyle}
        onSelectStyle={setSelectedStyle}
      />
      
      <TransformContainer>
        <UploadSection>
          {!previewUrl ? (
            <UploadBox
              isDragging={isDragging}
              onClick={handleUploadClick}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <UploadIcon />
              <UploadText>Glissez-déposez une image ici</UploadText>
              <UploadSubtext>ou cliquez pour parcourir</UploadSubtext>
              <HiddenInput
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="image/*"
              />
            </UploadBox>
          ) : (
            <PreviewContainer>
              <img src={previewUrl} alt="Aperçu" />
              <RemoveButton onClick={handleRemoveImage}>×</RemoveButton>
            </PreviewContainer>
          )}
          
          <TransformButton
            onClick={handleTransform}
            disabled={isLoading || !uploadedImage || !selectedStyle}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            {isLoading ? (
              <>
                <FaSpinner className="spin" /> Transformation en cours...
              </>
            ) : (
              <>
                <FaMagic /> Transformer l'image
              </>
            )}
          </TransformButton>
        </UploadSection>
        
        <ResultSection>
          {isLoading ? (
            <LoadingSpinner
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <FaSpinner className="spin" />
              <p>Transformation de votre image en cours...</p>
              <p>Cela peut prendre quelques instants</p>
            </LoadingSpinner>
          ) : generatedImage ? (
            <PreviewContainer>
              <img src={generatedImage} alt="Image transformée" />
            </PreviewContainer>
          ) : (
            <div className="text-center">
              <p>L'image transformée apparaîtra ici</p>
            </div>
          )}
        </ResultSection>
      </TransformContainer>
    </PageContainer>
  );
};

export default TransformPage;
