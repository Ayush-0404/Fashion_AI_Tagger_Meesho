
export interface PredictionResult {
  id: number;
  name: string;
  attributes: {
    color: string;
    pattern: string;
    category: string;
    sleeve: string;
    neckline: string;
    fit: string;
    occasion: string;
    material: string;
    season: string;
    style: string;
  };
  confidence: number;
  imageUrl: string;
}
