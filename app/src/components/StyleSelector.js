import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
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
  cursor: pointer;
  text-align: center;
  box-shadow: var(--box-shadow);
  border: 2px solid ${props => props.selected ? 'var(--primary-color)' : 'transparent'};
  transition: var(--transition);
  
  &:hover {
    transform: translateY(-5px);
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

// Animation variants
const cardVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 }
};

// Couleurs pour les styles
const styleColors = {
  disney: '#ff9d00',
  anime: '#ff5e7d',
  pixar: '#00c6ff',
  watercolor: '#7ed6df',
  comic: '#e056fd'
};

const StyleSelector = ({ selectedStyle, onSelectStyle }) => {
  const { availableStyles } = useImageContext();
  
  return (
    <StylesContainer>
      <StyleCard
        key="none"
        selected={selectedStyle === null}
        onClick={() => onSelectStyle(null)}
        as={motion.div}
        variants={cardVariants}
        initial="hidden"
        animate="visible"
        transition={{ duration: 0.3 }}
      >
        <StyleIcon color="#999">N</StyleIcon>
        <StyleName>Normal</StyleName>
        <StyleDescription>Image sans style particulier</StyleDescription>
      </StyleCard>
      
      {availableStyles.map((style, index) => (
        <StyleCard
          key={style.id}
          selected={selectedStyle === style.id}
          onClick={() => onSelectStyle(style.id)}
          as={motion.div}
          variants={cardVariants}
          initial="hidden"
          animate="visible"
          transition={{ duration: 0.3, delay: index * 0.1 }}
        >
          <StyleIcon color={styleColors[style.id] || '#6a11cb'}>
            {style.name.charAt(0)}
          </StyleIcon>
          <StyleName>{style.name}</StyleName>
          <StyleDescription>{style.description}</StyleDescription>
        </StyleCard>
      ))}
    </StylesContainer>
  );
};

export default StyleSelector;
