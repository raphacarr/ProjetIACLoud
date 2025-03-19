import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import styled from 'styled-components';
import { ImageProvider } from './context/ImageContext';
import GlobalStyles from './styles/GlobalStyles';
import Navbar from './components/Navbar';
import HomePage from './pages/HomePage';
import TransformPage from './pages/TransformPage';
import HistoryPage from './pages/HistoryPage';
import { checkApiHealth } from './services/api';

const AppContainer = styled.div`
  min-height: 100vh;
  display: flex;
  flex-direction: column;
`;

const Content = styled.main`
  flex: 1;
`;

const Footer = styled.footer`
  text-align: center;
  padding: 1.5rem;
  background: linear-gradient(to right, var(--primary-color), var(--secondary-color));
  color: white;
  margin-top: 2rem;
`;

const ApiStatusBanner = styled.div`
  background-color: ${props => props.isConnected ? 'var(--success-color)' : 'var(--error-color)'};
  color: white;
  text-align: center;
  padding: 0.5rem;
  font-size: 0.9rem;
`;

function App() {
  const [apiConnected, setApiConnected] = useState(false);
  const [apiChecked, setApiChecked] = useState(false);

  // Vérifier la connexion à l'API au chargement
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const health = await checkApiHealth();
        setApiConnected(health.status === 'healthy');
        if (health.status === 'healthy') {
          toast.success('Connexion à l\'API établie');
        } else {
          toast.error('Problème de connexion à l\'API');
        }
      } catch (error) {
        console.error('API connection error:', error);
        setApiConnected(false);
        toast.error('Impossible de se connecter à l\'API');
      } finally {
        setApiChecked(true);
      }
    };

    checkConnection();
  }, []);

  return (
    <Router>
      <ImageProvider>
        <AppContainer>
          <GlobalStyles />
          <ToastContainer position="top-right" autoClose={3000} />
          
          {apiChecked && !apiConnected && (
            <ApiStatusBanner isConnected={false}>
              ⚠️ L'API n'est pas accessible. Certaines fonctionnalités peuvent ne pas fonctionner correctement.
            </ApiStatusBanner>
          )}
          
          <Navbar />
          
          <Content>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/transform" element={<TransformPage />} />
              <Route path="/history" element={<HistoryPage />} />
            </Routes>
          </Content>
          
          <Footer>
            <p> {new Date().getFullYear()} DreamStyle Generator - Tous droits réservés</p>
          </Footer>
        </AppContainer>
      </ImageProvider>
    </Router>
  );
}

export default App;
