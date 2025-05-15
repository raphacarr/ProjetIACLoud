import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { FaTools } from 'react-icons/fa';
import { useImageContext } from '../context/ImageContext';

const StylesContainer = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 1rem;
  margin: 1.5rem 0;
`;

const StyleCard = styled(motion.div)`
  background: var(--card-background);
  border-radius: var(--border-radius);
  padding: 1rem;
  cursor: ${props => props.disabled ? 'not-allowed' : 'pointer'};
  text-align: center;
  box-shadow: var(--box-shadow);
  border: 2px solid ${props => props.selected ? 'var(--primary-color)' : 'transparent'};
  opacity: ${props => props.disabled ? 0.6 : 1};
  transition: var(--transition);
  
  &:hover {
    transform: ${props => !props.disabled && 'translateY(-5px)'};
  }
`;

const StyleIcon = styled.div`
  width: 80px;
  height: 80px;
  border-radius: 50%;
  background: ${props => `linear-gradient(45deg, var(--primary-color), ${props.color})`};
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 1rem;
  color: white;
  font-size: 2rem;
`;

const StyleName = styled.h3`
  font-size: 1rem;
  margin-bottom: 0.5rem;
`;

const StyleDescription = styled.p`
  font-size: 0.8rem;
  color: var(--text-secondary);
`;

const ComingSoonBadge = styled.div`
  background-color: var(--warning-color);
  color: white;
  font-size: 0.7rem;
  padding: 0.2rem 0.5rem;
  border-radius: 10px;
  margin-top: 0.5rem;
  display: inline-block;
`;

// Animation variants
const cardVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 }
};

// Fonction pour obtenir une icône pour un style donné
const getIconForStyle = (styleId) => {
  // Vous pouvez personnaliser cette fonction pour afficher des icônes spécifiques
  // Pour l'instant, on utilise simplement la première lettre du style
  return styleId.charAt(0).toUpperCase();
};

const StyleSelector = ({ selectedStyle, onSelectStyle }) => {
  const { availableStyles } = useImageContext();
  
  return (
    <StylesContainer>
      {availableStyles.map((style, index) => (
        <StyleCard
          key={style.id}
          selected={selectedStyle === style.id}
          onClick={() => style.available && onSelectStyle(style.id)}
          variants={cardVariants}
          initial="hidden"
          animate="visible"
          transition={{ duration: 0.3, delay: index * 0.1 }}
          disabled={!style.available}
        >
          <StyleIcon color={style.color}>
            {!style.available ? <FaTools /> : getIconForStyle(style.id)}
          </StyleIcon>
          <StyleName>{style.name}</StyleName>
          <StyleDescription>{style.description}</StyleDescription>
          {!style.available && (
            <ComingSoonBadge>En développement</ComingSoonBadge>
          )}
        </StyleCard>
      ))}
    </StylesContainer>
  );
};

export default StyleSelector;