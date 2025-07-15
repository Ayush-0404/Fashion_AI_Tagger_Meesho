
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Sparkles } from "lucide-react";
import { PredictionResult } from "@/types/prediction";

interface AttributesPanelProps {
  prediction: PredictionResult;
}

const AttributesPanel = ({ prediction }: AttributesPanelProps) => {
  return (
    <Card className="border-purple-200 shadow-xl rounded-2xl bg-white/90">
      <CardHeader>
        <CardTitle className="text-xl font-bold text-gray-800 flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-purple-600" />
          Predicted Attributes
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {Object.entries(prediction.attributes).map(([key, value]) => (
          <div key={key} className="flex justify-between items-center p-3 bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg">
            <span className="font-medium text-gray-700 capitalize">
              {key}:
            </span>
            <Badge className="bg-gradient-to-r from-purple-600 to-pink-600 text-white font-medium px-3 py-1 rounded-full shadow">
              {value}
            </Badge>
          </div>
        ))}
      </CardContent>
    </Card>
  );
};

export default AttributesPanel;
