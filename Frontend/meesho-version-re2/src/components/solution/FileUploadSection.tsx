
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Upload, Sparkles, X } from "lucide-react";
import { useState } from "react";

interface FileUploadSectionProps {
  uploadedImages: File[];
  onFileUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onAnalyze: () => void;
  onRemoveImage: (index: number) => void;
}

const FileUploadSection = ({ uploadedImages, onFileUpload, onAnalyze, onRemoveImage }: FileUploadSectionProps) => {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    console.log('File input changed');
    onFileUpload(event);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    const validFiles = files.filter(file => file.type.startsWith('image/'));
    
    if (validFiles.length > 10) {
      alert('Maximum 10 images allowed');
      return;
    }

    // Create a proper file input element and trigger the change event
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.multiple = true;
    
    // Create a DataTransfer object to set the files
    const dataTransfer = new DataTransfer();
    validFiles.forEach(file => dataTransfer.items.add(file));
    fileInput.files = dataTransfer.files;
    
    // Create a proper change event
    const changeEvent = new Event('change', { bubbles: true }) as any;
    Object.defineProperty(changeEvent, 'target', {
      value: fileInput,
      enumerable: true
    });
    
    onFileUpload(changeEvent);
  };

  return (
    <Card className="max-w-4xl mx-auto mb-8 border-purple-200 shadow-lg">
      <CardHeader className="text-center pb-4">
        <CardTitle className="text-2xl font-bold text-gray-800 flex items-center justify-center gap-2">
          <Sparkles className="h-6 w-6 text-purple-600" />
          Upload Fashion Images
        </CardTitle>
      </CardHeader>
      <CardContent className="p-6">
        <div 
          className={`border-2 border-dashed rounded-2xl p-6 text-center transition-all duration-300 ${
            isDragOver 
              ? 'border-purple-500 bg-purple-100' 
              : 'border-purple-300 bg-gradient-to-br from-purple-50 to-pink-50 hover:from-purple-100 hover:to-pink-100'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <Upload className="h-10 w-10 text-purple-500 mx-auto mb-3" />
          <div className="mb-4">
            <p className="text-lg text-gray-700 mb-2 font-medium">
              {isDragOver ? 'Drop your images here' : 'Choose images from your computer or drag & drop'}
            </p>
            <p className="text-sm text-gray-500">
              Supports JPG, PNG files. Maximum 10 images.
            </p>
          </div>
          
          <input
            type="file"
            multiple
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
            id="file-upload"
          />
          <label htmlFor="file-upload" className="cursor-pointer">
            <Button 
              type="button"
              className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white px-6 py-2 rounded-full font-semibold shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-105"
              asChild
            >
              <span>Choose Files</span>
            </Button>
          </label>

          {/* Analyze Button - positioned right after upload area */}
          {uploadedImages.length > 0 && (
            <div className="mt-4 pt-4 border-t border-purple-200">
              <Button
                onClick={onAnalyze}
                disabled={uploadedImages.length === 0}
                className="bg-gradient-to-r from-pink-600 to-purple-600 hover:from-pink-700 hover:to-purple-700 text-white px-8 py-3 rounded-full text-lg font-bold shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
              >
                <Sparkles className="h-5 w-5 mr-2" />
                Analyze Images ({uploadedImages.length})
              </Button>
            </div>
          )}
        </div>

        {uploadedImages.length > 0 && (
          <div className="mt-6">
            <h3 className="font-semibold text-gray-800 mb-3 text-lg">
              Selected Images ({uploadedImages.length}/10)
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              {uploadedImages.map((file, index) => (
                <div key={index} className="relative group">
                  <img
                    src={URL.createObjectURL(file)}
                    alt={`Upload ${index + 1}`}
                    className="w-full h-32 object-cover rounded-lg shadow-md group-hover:shadow-lg transition-shadow duration-300"
                  />
                  <button
                    onClick={() => onRemoveImage(index)}
                    className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 opacity-0 group-hover:opacity-100 transition-opacity duration-200 hover:bg-red-600"
                  >
                    <X className="h-4 w-4" />
                  </button>
                  <div className="absolute bottom-2 left-2 bg-black/70 text-white text-xs px-2 py-1 rounded backdrop-blur-sm">
                    {file.name.length > 12 ? file.name.substring(0, 12) + '...' : file.name}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default FileUploadSection;
