import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { useImageContext } from '../context/ImageContext';
import ImageCard from '../components/ImageCard';

const PageContainer = styled.div`
  max-width: 1200px;
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

const FiltersContainer = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
  flex-wrap: wrap;
  gap: 1rem;
`;

const SearchInput = styled.input`
  padding: 0.8rem;
  border-radius: var(--border-radius);
  border: 1px solid #ddd;
  width: 100%;
  max-width: 300px;
  
  &:focus {
    border-color: var(--primary-color);
    box-shadow: 0 0 0 2px rgba(106, 17, 203, 0.2);
  }
`;

const FilterSelect = styled.select`
  padding: 0.8rem;
  border-radius: var(--border-radius);
  border: 1px solid #ddd;
  background: white;
  
  &:focus {
    border-color: var(--primary-color);
    outline: none;
  }
`;

const ImagesGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 2rem;
`;

const EmptyState = styled.div`
  text-align: center;
  padding: 3rem;
  background: white;
  border-radius: var(--border-radius);
  box-shadow: var(--box-shadow);
`;

const EmptyStateTitle = styled.h3`
  font-size: 1.5rem;
  margin-bottom: 1rem;
  color: var(--text-primary);
`;

const EmptyStateText = styled.p`
  color: var(--text-secondary);
  margin-bottom: 1.5rem;
`;

const EmptyStateButton = styled.button`
  padding: 0.8rem 1.5rem;
  background: linear-gradient(to right, var(--primary-color), var(--secondary-color));
  color: white;
  border: none;
  border-radius: var(--border-radius);
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
  }
`;

// Animation variants
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
};

const HistoryPage = () => {
  const { imageHistory } = useImageContext();
  const [searchTerm, setSearchTerm] = React.useState('');
  const [styleFilter, setStyleFilter] = React.useState('all');
  
  // Filtrer les images en fonction des critères de recherche et du filtre de style
  const filteredImages = React.useMemo(() => {
    return imageHistory.filter(image => {
      const matchesSearch = image.prompt.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesStyle = styleFilter === 'all' || image.style === styleFilter;
      return matchesSearch && matchesStyle;
    });
  }, [imageHistory, searchTerm, styleFilter]);
  
  // Extraire les styles uniques pour le filtre
  const uniqueStyles = React.useMemo(() => {
    const styles = imageHistory.map(image => image.style).filter(Boolean);
    return ['all', ...new Set(styles)];
  }, [imageHistory]);
  
  return (
    <PageContainer>
      <PageTitle
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        Historique des Images
      </PageTitle>
      <PageDescription>
        Retrouvez toutes les images que vous avez générées et transformées.
      </PageDescription>
      
      <FiltersContainer>
        <SearchInput
          type="text"
          placeholder="Rechercher par prompt..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
        
        <FilterSelect
          value={styleFilter}
          onChange={(e) => setStyleFilter(e.target.value)}
        >
          {uniqueStyles.map(style => (
            <option key={style} value={style}>
              {style === 'all' ? 'Tous les styles' : style}
            </option>
          ))}
        </FilterSelect>
      </FiltersContainer>
      
      {filteredImages.length > 0 ? (
        <ImagesGrid
          as={motion.div}
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          {filteredImages.map(image => (
            <ImageCard key={image.id} image={image} />
          ))}
        </ImagesGrid>
      ) : (
        <EmptyState>
          <EmptyStateTitle>Aucune image trouvée</EmptyStateTitle>
          <EmptyStateText>
            {imageHistory.length === 0
              ? "Vous n'avez pas encore généré ou transformé d'images."
              : "Aucune image ne correspond à vos critères de recherche."}
          </EmptyStateText>
          <EmptyStateButton onClick={() => window.location.href = '/'}>
            Générer une image
          </EmptyStateButton>
        </EmptyState>
      )}
    </PageContainer>
  );
};

export default HistoryPage;
