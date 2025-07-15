
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { PredictionResult } from "@/types/prediction";

interface ImagePreviewProps {
  prediction: PredictionResult;
  predictions: PredictionResult[];
  selectedImageIndex: number;
  onImageSelect: (index: number) => void;
}

const ImagePreview = ({ prediction, predictions, selectedImageIndex, onImageSelect }: ImagePreviewProps) => {
  const handlePrevious = () => {
    const newIndex = selectedImageIndex > 0 ? selectedImageIndex - 1 : predictions.length - 1;
    onImageSelect(newIndex);
  };

  const handleNext = () => {
    const newIndex = selectedImageIndex < predictions.length - 1 ? selectedImageIndex + 1 : 0;
    onImageSelect(newIndex);
  };

  return (
    <div className="lg:col-span-2 space-y-6">
      {/* Main Image with Navigation */}
      <Card className="border-purple-200 shadow-lg overflow-hidden">
        <CardContent className="p-0 relative">
          <div className="flex items-center justify-center bg-gray-50 min-h-96">
            <img
              src={prediction.imageUrl}
              alt={`Analysis result ${selectedImageIndex + 1}`}
              className="max-w-full max-h-96 object-contain rounded-xl shadow-lg ring-2 ring-purple-200"
            />
          </div>
          
          {/* Navigation Arrows */}
          {predictions.length > 1 && (
            <>
              <Button
                onClick={handlePrevious}
                variant="outline"
                size="icon"
                className="absolute left-4 top-1/2 transform -translate-y-1/2 bg-white/90 hover:bg-white border-purple-200 shadow-lg"
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <Button
                onClick={handleNext}
                variant="outline"
                size="icon"
                className="absolute right-4 top-1/2 transform -translate-y-1/2 bg-white/90 hover:bg-white border-purple-200 shadow-lg"
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </>
          )}
          
          {/* Image Info */}
          <div className="p-4 bg-gradient-to-r from-purple-50 to-pink-50">
            <h3 className="font-semibold text-gray-800 mb-2">
              {prediction.name}
            </h3>
            <div className="flex items-center justify-between">
              <div className="flex items-center flex-1">
                <span className="text-sm text-gray-600 mr-2">Confidence:</span>
                <div className="flex-1 bg-purple-200 rounded-full h-2 mr-3">
                  <div
                    className="bg-gradient-to-r from-purple-600 to-pink-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${prediction.confidence * 100}%` }}
                  ></div>
                </div>
                <span className="text-sm font-bold text-purple-600">
                  {(prediction.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <span className="text-sm text-gray-500 ml-4">
                {selectedImageIndex + 1} of {predictions.length}
              </span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Thumbnail Navigation - Full Width */}
      {predictions.length > 1 && (
        <Card className="border-purple-200 shadow-lg col-span-full">
          <CardContent className="p-4">
            <h4 className="font-semibold text-gray-800 mb-3">Browse Images</h4>
            <div className="grid grid-cols-5 md:grid-cols-9 gap-3">
              {predictions.map((pred, index) => (
                <button
                  key={index}
                  onClick={() => onImageSelect(index)}
                  className={`relative rounded-lg overflow-hidden transition-all duration-300 aspect-square ${
                    selectedImageIndex === index 
                      ? 'ring-3 ring-purple-500 shadow-lg scale-105' 
                      : 'hover:ring-2 hover:ring-purple-300 hover:scale-102'
                  }`}
                >
                  <img
                    src={pred.imageUrl}
                    alt={`Thumbnail ${index + 1}`}
                    className="w-full h-full object-cover rounded-lg shadow-md group-hover:shadow-lg transition-shadow duration-300 ring-2 ring-purple-100"
                  />
                  <div className="absolute inset-0 bg-black/20 flex items-center justify-center">
                    <span className="text-white font-bold text-sm">{index + 1}</span>
                  </div>
                </button>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default ImagePreview;
