
import { useState, useCallback } from 'react';
import { PredictionResult } from '@/types/prediction';
import { generateMockPredictions, downloadCSV, downloadPDF } from '@/utils/predictionService';
import SolutionNavigation from '@/components/solution/SolutionNavigation';
import SolutionHero from '@/components/solution/SolutionHero';
import FileUploadSection from '@/components/solution/FileUploadSection';
import LoadingAnimation from '@/components/solution/LoadingAnimation';
import ResultsSection from '@/components/solution/ResultsSection';
import jsPDF from "jspdf";

//const BACKEND_URL = "https://fashion-ai-tagger-meesho.onrender.com";
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL;

const Solution = () => {
  const [uploadedImages, setUploadedImages] = useState<File[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [selectedImageIndex, setSelectedImageIndex] = useState(0);

  const handleFileUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    console.log('File upload triggered');
    const files = Array.from(event.target.files || []);
    console.log('Files selected:', files.length);
    
    const validFiles = files.filter(file => file.type.startsWith('image/'));
    console.log('Valid image files:', validFiles.length);
    
    if (validFiles.length > 10) {
      alert('Maximum 10 images allowed');
      return;
    }
    
    setUploadedImages(prev => {
      const newImages = [...prev, ...validFiles].slice(0, 10);
      console.log('Updated uploaded images:', newImages.length);
      return newImages;
    });
    setPredictions([]);
    setSelectedImageIndex(0);
    
    // Clear the input value to allow selecting the same files again
    event.target.value = '';
  }, []);

  const handleRemoveImage = useCallback((index: number) => {
    setUploadedImages(prev => prev.filter((_, i) => i !== index));
    setPredictions([]);
    setSelectedImageIndex(0);
  }, []);

  const analyzeImages = async () => {
    console.log('Analyze button clicked, images:', uploadedImages.length);
    if (uploadedImages.length === 0) return;
    
    setIsProcessing(true);
    
    const formData = new FormData();
    uploadedImages.forEach((file) => {
      formData.append("files", file);
    });
    
    try {
      const response = await fetch(`${BACKEND_URL}/predict`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error("Failed to trigger backend prediction");
      }
      // Wait for backend to finish and then fetch predictions.json
      const predResponse = await fetch(`${BACKEND_URL}/predictions`);
      if (!predResponse.ok) {
        throw new Error("Failed to fetch predictions from backend");
      }
      const results = await predResponse.json();
      // results is an object: { filename: { category, attributes } }
      // Convert to array for UI
      const predictionsArr = Object.entries(results).map(([filename, data], idx) => {
        const attrs = (data as { attributes: any }).attributes;
        return {
          id: idx,
          name: filename,
          attributes: attrs,
          imageUrl: `${BACKEND_URL}/images/${filename}`,
          confidence: 1 // or set to a real value if available
        };
      });
      setPredictions(predictionsArr);
      setSelectedImageIndex(0);
    } catch (error) {
      alert("Error analyzing images. Please try again.");
      console.error(error);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownloadCSV = () => {
    downloadCSV(predictions);
  };

  const handleDownloadPDF = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/download-pdf`);
      if (!response.ok) throw new Error("Failed to download PDF");
      const blob = await response.blob();
      // Create a link and trigger download
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "predictions_report.pdf";
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      alert("Failed to download PDF.");
      console.error(err);
    }
  };

  const handleStartNew = () => {
    setUploadedImages([]);
    setPredictions([]);
    setSelectedImageIndex(0);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-white">
      <SolutionNavigation />
      
      <div className="container mx-auto px-6 py-12">
        <SolutionHero />

        {!predictions.length && !isProcessing && (
          <FileUploadSection
            uploadedImages={uploadedImages}
            onFileUpload={handleFileUpload}
            onAnalyze={analyzeImages}
            onRemoveImage={handleRemoveImage}
          />
        )}

        {isProcessing && <LoadingAnimation />}

        {predictions.length > 0 && !isProcessing && (
          <ResultsSection
            predictions={predictions}
            selectedImageIndex={selectedImageIndex}
            onImageSelect={setSelectedImageIndex}
            onDownloadPDF={handleDownloadPDF}
            onDownloadCSV={handleDownloadCSV}
            onStartNew={handleStartNew}
          />
        )}
      </div>
    </div>
  );
};

export default Solution;
