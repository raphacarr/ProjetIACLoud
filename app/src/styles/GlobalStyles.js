import { createGlobalStyle } from 'styled-components';

const GlobalStyles = createGlobalStyle`
  :root {
    --primary-color: #6a11cb;
    --secondary-color: #2575fc;
    --accent-color: #ff9d00;
    --background-color: #f5f7fa;
    --card-background: #ffffff;
    --text-primary: #333333;
    --text-secondary: #666666;
    --text-light: #999999;
    --success-color: #28a745;
    --error-color: #dc3545;
    --warning-color: #ffc107;
    --border-radius: 8px;
    --box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    --transition: all 0.3s ease;
  }

  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  body {
    font-family: 'Poppins', sans-serif;
    background: var(--background-color);
    color: var(--text-primary);
    line-height: 1.6;
  }

  a {
    text-decoration: none;
    color: var(--primary-color);
    transition: var(--transition);
    
    &:hover {
      color: var(--secondary-color);
    }
  }

  button {
    cursor: pointer;
    border: none;
    border-radius: var(--border-radius);
    padding: 10px 20px;
    font-weight: 600;
    transition: var(--transition);
    background: linear-gradient(to right, var(--primary-color), var(--secondary-color));
    color: white;
    
    &:hover {
      transform: translateY(-2px);
      box-shadow: var(--box-shadow);
    }
    
    &:disabled {
      opacity: 0.7;
      cursor: not-allowed;
      transform: none;
    }
  }

  input, textarea, select {
    width: 100%;
    padding: 12px;
    border: 1px solid #ddd;
    border-radius: var(--border-radius);
    font-family: inherit;
    transition: var(--transition);
    
    &:focus {
      outline: none;
      border-color: var(--primary-color);
      box-shadow: 0 0 0 2px rgba(106, 17, 203, 0.2);
    }
  }

  h1, h2, h3, h4, h5, h6 {
    margin-bottom: 1rem;
    font-weight: 700;
    line-height: 1.2;
  }

  .container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
  }

  .card {
    background: var(--card-background);
    border-radius: var(--border-radius);
    box-shadow: var(--box-shadow);
    padding: 20px;
    transition: var(--transition);
    
    &:hover {
      transform: translateY(-5px);
      box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }
  }

  .flex {
    display: flex;
  }

  .flex-center {
    display: flex;
    justify-content: center;
    align-items: center;
  }

  .flex-between {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .grid {
    display: grid;
    gap: 20px;
  }

  .btn-secondary {
    background: white;
    color: var(--primary-color);
    border: 1px solid var(--primary-color);
    
    &:hover {
      background: rgba(106, 17, 203, 0.1);
    }
  }

  .btn-accent {
    background: var(--accent-color);
    
    &:hover {
      background: darken(var(--accent-color), 10%);
    }
  }

  .text-center {
    text-align: center;
  }

  .mb-1 { margin-bottom: 0.5rem; }
  .mb-2 { margin-bottom: 1rem; }
  .mb-3 { margin-bottom: 1.5rem; }
  .mb-4 { margin-bottom: 2rem; }
  .mt-1 { margin-top: 0.5rem; }
  .mt-2 { margin-top: 1rem; }
  .mt-3 { margin-top: 1.5rem; }
  .mt-4 { margin-top: 2rem; }
`;

export default GlobalStyles;
