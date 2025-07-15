
import { PredictionResult } from "@/types/prediction";

export const generateMockPredictions = (uploadedImages: File[]): PredictionResult[] => {
  const colors = ['Navy Blue', 'Crimson Red', 'Emerald Green', 'Golden Yellow', 'Royal Purple'];
  const patterns = ['Polka Dot', 'Striped', 'Floral', 'Geometric', 'Solid'];
  const categories = ['Dress', 'T-shirt', 'Blazer', 'Skirt', 'Jeans'];
  const sleeves = ['Full Sleeve', 'Half Sleeve', 'Sleeveless', '3/4 Sleeve', 'Cap Sleeve'];
  const necklines = ['Round Neck', 'V-Neck', 'Boat Neck', 'High Neck', 'Off Shoulder'];
  const fits = ['Regular Fit', 'Slim Fit', 'Loose Fit', 'Tailored', 'Oversized'];
  const occasions = ['Casual', 'Formal', 'Party', 'Office', 'Weekend'];
  const materials = ['Cotton', 'Silk', 'Polyester', 'Chiffon', 'Denim'];
  const seasons = ['Summer', 'Winter', 'Spring', 'Monsoon', 'All Season'];
  const styles = ['Contemporary', 'Traditional', 'Boho', 'Minimalist', 'Vintage'];

  return uploadedImages.map((file, index) => ({
    id: index,
    name: file.name,
    attributes: {
      color: colors[Math.floor(Math.random() * colors.length)],
      pattern: patterns[Math.floor(Math.random() * patterns.length)],
      category: categories[Math.floor(Math.random() * categories.length)],
      sleeve: sleeves[Math.floor(Math.random() * sleeves.length)],
      neckline: necklines[Math.floor(Math.random() * necklines.length)],
      fit: fits[Math.floor(Math.random() * fits.length)],
      occasion: occasions[Math.floor(Math.random() * occasions.length)],
      material: materials[Math.floor(Math.random() * materials.length)],
      season: seasons[Math.floor(Math.random() * seasons.length)],
      style: styles[Math.floor(Math.random() * styles.length)]
    },
    confidence: 0.85 + Math.random() * 0.15,
    imageUrl: URL.createObjectURL(file)
  }));
};

export const downloadCSV = (predictions: PredictionResult[]) => {
  const csvContent = [
    'Image Name,Color,Pattern,Category,Sleeve,Neckline,Fit,Occasion,Material,Season,Style,Confidence',
    ...predictions.map(p => 
      `${p.name},${p.attributes.color},${p.attributes.pattern},${p.attributes.category},${p.attributes.sleeve},${p.attributes.neckline},${p.attributes.fit},${p.attributes.occasion},${p.attributes.material},${p.attributes.season},${p.attributes.style},${(p.confidence * 100).toFixed(1)}%`
    )
  ].join('\n');
  
  const blob = new Blob([csvContent], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'fashion-predictions.csv';
  a.click();
};

export const downloadPDF = () => {
  // Simulate PDF download
  alert('PDF report would be generated and downloaded here');
};
