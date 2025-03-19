import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import styled from 'styled-components';
import { FaPaintBrush, FaHistory, FaHome } from 'react-icons/fa';

const NavbarContainer = styled.nav`
  background: linear-gradient(to right, var(--primary-color), var(--secondary-color));
  padding: 1rem 2rem;
  color: white;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  position: sticky;
  top: 0;
  z-index: 100;
`;

const NavbarContent = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  max-width: 1200px;
  margin: 0 auto;
`;

const Logo = styled.div`
  font-size: 1.8rem;
  font-weight: 700;
  display: flex;
  align-items: center;
  gap: 10px;
  
  a {
    color: white;
    text-decoration: none;
  }
`;

const NavLinks = styled.div`
  display: flex;
  gap: 2rem;
`;

const NavLink = styled(Link)`
  color: white;
  text-decoration: none;
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 0.5rem 1rem;
  border-radius: var(--border-radius);
  transition: var(--transition);
  font-weight: ${props => props.active ? '700' : '400'};
  background: ${props => props.active ? 'rgba(255, 255, 255, 0.2)' : 'transparent'};
  
  &:hover {
    background: rgba(255, 255, 255, 0.2);
  }
  
  svg {
    font-size: 1.2rem;
  }
`;

const Navbar = () => {
  const location = useLocation();
  
  const isActive = (path) => {
    return location.pathname === path;
  };
  
  return (
    <NavbarContainer>
      <NavbarContent>
        <Logo>
          <Link to="/">
            <FaPaintBrush /> DreamStyle
          </Link>
        </Logo>
        <NavLinks>
          <NavLink to="/" active={isActive('/')}>
            <FaHome /> Accueil
          </NavLink>
          <NavLink to="/transform" active={isActive('/transform')}>
            <FaPaintBrush /> Transformer
          </NavLink>
          <NavLink to="/history" active={isActive('/history')}>
            <FaHistory /> Historique
          </NavLink>
        </NavLinks>
      </NavbarContent>
    </NavbarContainer>
  );
};

export default Navbar;
