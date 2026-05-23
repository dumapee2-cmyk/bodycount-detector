import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './globals.css';
import FaceAnalysis from './FaceAnalysis';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <FaceAnalysis />
  </StrictMode>,
);
