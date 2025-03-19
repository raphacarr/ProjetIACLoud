import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { FaDownload, FaShare, FaInfoCircle } from 'react-icons/fa';

const Card = styled(motion.div)`
  background: var(--card-background);
  border-radius: var(--border-radius);
  overflow: hidden;
  box-shadow: var(--box-shadow);
  transition: var(--transition);
  
  &:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
  }
`;

const ImageContainer = styled.div`
  position: relative;
  width: 100%;
  height: 0;
  padding-bottom: 100%; /* Aspect ratio 1:1 */
  overflow: hidden;
`;

const StyledImage = styled.img`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.3s ease;
  
  ${Card}:hover & {
    transform: scale(1.05);
  }
`;

const ImageInfo = styled.div`
  padding: 1rem;
`;

const ImagePrompt = styled.p`
  font-size: 0.9rem;
  color: var(--text-primary);
  margin-bottom: 0.5rem;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  text-overflow: ellipsis;
`;

const ImageMeta = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 0.5rem;
`;

const StyleBadge = styled.span`
  background: linear-gradient(to right, var(--primary-color), var(--secondary-color));
  color: white;
  padding: 0.2rem 0.5rem;
  border-radius: 20px;
  font-size: 0.7rem;
  font-weight: 600;
`;

const ImageDate = styled.span`
  font-size: 0.7rem;
  color: var(--text-light);
`;

const ActionButtons = styled.div`
  display: flex;
  gap: 0.5rem;
  margin-top: 0.5rem;
`;

const ActionButton = styled.button`
  background: transparent;
  color: var(--text-secondary);
  padding: 0.5rem;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: var(--transition);
  
  &:hover {
    background: rgba(0, 0, 0, 0.05);
    color: var(--primary-color);
    transform: translateY(0);
    box-shadow: none;
  }
`;

const ImageCard = ({ image }) => {
  const { prompt, style, imageUrl, createdAt } = image;
  
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('fr-FR', {
      day: 'numeric',
      month: 'short',
      year: 'numeric',
    });
  };
  
  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = imageUrl;
    link.download = `dreamstyle-${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  
  return (
    <Card
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <ImageContainer>
        <StyledImage src={imageUrl} alt={prompt} />
      </ImageContainer>
      <ImageInfo>
        <ImagePrompt>{prompt}</ImagePrompt>
        <ImageMeta>
          {style && <StyleBadge>{style}</StyleBadge>}
          <ImageDate>{formatDate(createdAt)}</ImageDate>
        </ImageMeta>
        <ActionButtons>
          <ActionButton onClick={handleDownload} title="Télécharger">
            <FaDownload />
          </ActionButton>
          <ActionButton title="Partager">
            <FaShare />
          </ActionButton>
          <ActionButton title="Détails">
            <FaInfoCircle />
          </ActionButton>
        </ActionButtons>
      </ImageInfo>
    </Card>
  );
};

export default ImageCard;
