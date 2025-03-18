import React, { useState } from 'react';

function App() {
  const [prompt, setPrompt] = useState("");
  const [generatedImage, setGeneratedImage] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleGenerate = async () => {
    setLoading(true);
    setGeneratedImage(null);
    try {
      const response = await fetch("https://ton-api-gateway-url.com/prod/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt })
      });
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setGeneratedImage(url);
    } catch (err) {
      console.error(err);
      alert("Erreur lors de la génération");
    }
    setLoading(false);
  };

  return (
    <div>
      <h1>Génération d'images DreamBooth</h1>
      <input 
        type="text" 
        value={prompt} 
        onChange={(e) => setPrompt(e.target.value)} 
        placeholder="Prompt..." 
      />
      <button onClick={handleGenerate} disabled={loading}>
        {loading ? "Génération en cours..." : "Générer"}
      </button>
      {generatedImage && <img src={generatedImage} alt="Résultat" />}
    </div>
  );
}

export default App;
