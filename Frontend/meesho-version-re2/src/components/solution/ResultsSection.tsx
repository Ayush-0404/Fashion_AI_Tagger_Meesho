
import { PredictionResult } from "@/types/prediction";
import ImagePreview from "./ImagePreview";
import AttributesPanel from "./AttributesPanel";
import { Button } from "@/components/ui/button";
import { FileText, Download } from "lucide-react";

interface ResultsSectionProps {
  predictions: PredictionResult[];
  selectedImageIndex: number;
  onImageSelect: (index: number) => void;
  onDownloadPDF: () => void;
  onDownloadCSV: () => void;
  onStartNew: () => void;
}

const ResultsSection = ({ 
  predictions, 
  selectedImageIndex, 
  onImageSelect, 
  onDownloadPDF, 
  onDownloadCSV, 
  onStartNew 
}: ResultsSectionProps) => {
  return (
    <div className="max-w-7xl mx-auto">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent mb-4">
          Analysis Results
        </h2>
        <p className="text-gray-600">AI-powered fashion attribute prediction completed</p>
      </div>

      <hr className="my-8 border-t border-purple-100" />

      {/* Action Buttons at Top */}
      <div className="flex justify-center gap-4 mb-8">
        <Button
          onClick={onStartNew}
          variant="outline"
          className="border-purple-300 text-purple-600 hover:bg-purple-50 rounded-lg transition-all duration-200 hover:scale-105 hover:shadow-lg"
        >
          Start New Analysis
        </Button>
        <Button
          onClick={onDownloadPDF}
          className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white rounded-lg transition-all duration-200 hover:scale-105 hover:shadow-lg"
        >
          <FileText className="h-4 w-4 mr-2" />
          Download Report (PDF)
        </Button>
        <Button
          onClick={onDownloadCSV}
          variant="outline"
          className="border-purple-300 text-purple-600 hover:bg-purple-50 rounded-lg transition-all duration-200 hover:scale-105 hover:shadow-lg"
        >
          <Download className="h-4 w-4 mr-2" />
          Download CSV
        </Button>
      </div>

      <div className="grid lg:grid-cols-3 gap-8">
        <ImagePreview
          prediction={predictions[selectedImageIndex]}
          predictions={predictions}
          selectedImageIndex={selectedImageIndex}
          onImageSelect={onImageSelect}
        />
        
        <AttributesPanel
          prediction={predictions[selectedImageIndex]}
        />
      </div>
    </div>
  );
};

export default ResultsSection;
