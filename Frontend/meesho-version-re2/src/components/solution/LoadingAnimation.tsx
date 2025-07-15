
import { Card, CardContent } from "@/components/ui/card";
import { Brain, Zap, Eye, Sparkles, Star, Heart, Diamond } from "lucide-react";

const LoadingAnimation = () => {
  return (
    <div className="fixed inset-0 z-50 bg-black/20 backdrop-blur-md flex items-center justify-center">
      {/* Moving Characters */}
      <div className="absolute inset-0 overflow-hidden">
        {/* Character 1 - Brain moving left to right */}
        <div className="absolute top-1/4 -left-16 animate-[slide-right_4s_ease-in-out_infinite]">
          <Brain className="h-12 w-12 text-purple-500" />
        </div>
        
        {/* Character 2 - Zap moving right to left */}
        <div className="absolute top-1/3 -right-16 animate-[slide-left_3s_ease-in-out_infinite] [animation-delay:0.5s]">
          <Zap className="h-10 w-10 text-pink-500" />
        </div>
        
        {/* Character 3 - Eye moving diagonally */}
        <div className="absolute top-3/4 -left-16 animate-[slide-diagonal_5s_ease-in-out_infinite] [animation-delay:1s]">
          <Eye className="h-14 w-14 text-purple-600" />
        </div>
        
        {/* Character 4 - Another Brain */}
        <div className="absolute top-1/2 -right-16 animate-[slide-left_3.5s_ease-in-out_infinite] [animation-delay:1.5s]">
          <Brain className="h-8 w-8 text-pink-400" />
        </div>
      </div>

      {/* Central Content Card */}
      <Card className="max-w-2xl mx-auto border-purple-200 shadow-2xl bg-white/95 backdrop-blur-sm">
        <CardContent className="p-12 text-center relative overflow-hidden">
          {/* Floating Infographic Elements */}
          <div className="absolute inset-0 pointer-events-none">
            {/* Sparkles */}
            <Sparkles className="absolute top-4 left-8 h-6 w-6 text-yellow-400 animate-bounce [animation-delay:0s]" />
            <Star className="absolute top-6 right-12 h-5 w-5 text-pink-400 animate-bounce [animation-delay:0.3s]" />
            <Diamond className="absolute bottom-8 left-6 h-4 w-4 text-purple-400 animate-bounce [animation-delay:0.6s]" />
            <Heart className="absolute bottom-6 right-8 h-5 w-5 text-pink-500 animate-bounce [animation-delay:0.9s]" />
            
            {/* Floating circles */}
            <div className="absolute top-1/4 left-1/4 w-3 h-3 bg-purple-300 rounded-full animate-[float_3s_ease-in-out_infinite] [animation-delay:0s]"></div>
            <div className="absolute top-3/4 right-1/4 w-2 h-2 bg-pink-300 rounded-full animate-[float_3s_ease-in-out_infinite] [animation-delay:1s]"></div>
            <div className="absolute top-1/2 right-1/3 w-4 h-4 bg-purple-200 rounded-full animate-[float_3s_ease-in-out_infinite] [animation-delay:2s]"></div>
          </div>

          {/* Main Content */}
          <div className="relative z-10">
            {/* Central Icon Animation */}
            <div className="flex justify-center items-center mb-8">
              <div className="relative">
                <Brain className="h-20 w-20 text-purple-600 animate-pulse" />
                <div className="absolute inset-0 animate-ping">
                  <Brain className="h-20 w-20 text-purple-400 opacity-30" />
                </div>
              </div>
            </div>
            
            <h3 className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent mb-4">
              AI Analysis in Progress
            </h3>
            <p className="text-gray-600 text-lg mb-8">
              Our advanced AI is examining your fashion images and identifying their unique attributes...
            </p>
            
            {/* Animated Progress Indicators */}
            <div className="space-y-4">
              <div className="flex items-center justify-center gap-2 text-sm text-gray-500">
                <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse"></div>
                <span>Detecting colors and patterns</span>
              </div>
              <div className="flex items-center justify-center gap-2 text-sm text-gray-500 [animation-delay:1s]">
                <div className="w-2 h-2 bg-pink-500 rounded-full animate-pulse [animation-delay:1s]"></div>
                <span>Analyzing fabric and texture</span>
              </div>
              <div className="flex items-center justify-center gap-2 text-sm text-gray-500 [animation-delay:2s]">
                <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse [animation-delay:2s]"></div>
                <span>Identifying style attributes</span>
              </div>
            </div>
            
            {/* Progress Bar */}
            <div className="mt-8 w-80 mx-auto bg-purple-100 rounded-full h-3 relative overflow-hidden">
              <div className="bg-gradient-to-r from-purple-600 to-pink-600 h-3 rounded-full animate-[progress_4s_ease-in-out_infinite]"></div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Custom CSS animations */}
      <style>{`
        @keyframes slide-right {
          0% { transform: translateX(-100px); }
          50% { transform: translateX(calc(100vw + 100px)); }
          100% { transform: translateX(-100px); }
        }
        
        @keyframes slide-left {
          0% { transform: translateX(100px); }
          50% { transform: translateX(calc(-100vw - 100px)); }
          100% { transform: translateX(100px); }
        }
        
        @keyframes slide-diagonal {
          0% { transform: translate(-100px, 0); }
          50% { transform: translate(calc(100vw + 100px), -200px); }
          100% { transform: translate(-100px, 0); }
        }
        
        @keyframes progress {
          0% { width: 20%; }
          50% { width: 80%; }
          100% { width: 95%; }
        }
        
        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-20px); }
        }
      `}</style>
    </div>
  );
};

export default LoadingAnimation;
