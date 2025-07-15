import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ArrowRight, Clock, Users, Zap, Target, TrendingUp, Layers, Sparkles, CheckCircle } from "lucide-react";
import { Link } from 'react-router-dom';
const About = () => {
  return <div className="min-h-screen bg-white">
      {/* Navigation */}
      <nav className="fixed top-0 w-full bg-white/95 backdrop-blur-md z-50 border-b border-gray-100 shadow-sm">
        <div className="container mx-auto px-6 py-4 flex justify-between items-center">
          <Link to="/" className="font-heading font-bold text-2xl text-meesho-purple">
            Fashion AI Tagger
          </Link>
          <div className="flex space-x-6">
            
            <Link to="/solution" className="bg-meesho-orange text-white px-6 py-2 rounded-full font-semibold hover:bg-orange-500 transition-all duration-300 transform hover:scale-105">
              Try Now
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-24 pb-16 bg-gradient-meesho-hero text-white relative overflow-hidden">
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-20 left-10 w-16 h-16 bg-white/10 rounded-lg floating-elements"></div>
          <div className="absolute top-32 right-20 w-12 h-12 bg-meesho-orange/20 rounded-xl floating-elements" style={{
          animationDelay: '1s'
        }}></div>
          <div className="absolute bottom-20 left-32 w-20 h-20 bg-meesho-cyan/20 rounded-2xl floating-elements" style={{
          animationDelay: '2s'
        }}></div>
        </div>
        
        <div className="container mx-auto px-6 text-center relative z-10">
          <div className="mb-8">
            <span className="inline-flex items-center px-4 py-2 bg-white/20 backdrop-blur-sm rounded-full text-white text-sm font-medium mb-6">
              <Target className="w-4 h-4 mr-2" />
              Problem Statement
            </span>
          </div>
          
          <h1 className="font-heading text-5xl md:text-7xl font-bold mb-8 leading-tight">
            The Fashion Attribution
            <span className="block mt-2 text-gradient-meesho bg-clip-text text-transparent bg-gradient-to-r from-white via-meesho-orange to-white">
              Challenge
            </span>
          </h1>
          
          <p className="text-xl md:text-2xl text-white/90 mb-12 max-w-4xl mx-auto leading-relaxed">
            Understanding why manual fashion tagging is a billion-dollar bottleneck and how AI transforms this landscape
          </p>
        </div>
      </section>

      {/* Problem Scale Visualization */}
      <section className="py-24 bg-gray-50">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="font-heading text-4xl md:text-5xl font-bold text-meesho-purple mb-6">
              The Scale of the Problem
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Fashion e-commerce operates at unprecedented scale, creating attribution challenges that manual processes simply cannot handle
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto mb-16">
            {[{
            icon: <Layers className="h-12 w-12" />,
            number: "5M+",
            title: "Product SKUs",
            description: "Active fashion products requiring accurate attribute tagging across major platforms",
            color: "from-meesho-purple to-meesho-magenta"
          }, {
            icon: <Clock className="h-12 w-12" />,
            number: "10min",
            title: "Per Product",
            description: "Average time for manual tagging by trained annotators, including quality checks",
            color: "from-meesho-orange to-meesho-magenta"
          }, {
            icon: <Users className="h-12 w-12" />,
            number: "24/7",
            title: "Continuous Scale",
            description: "New products added every minute, creating an endless backlog of untagged items",
            color: "from-meesho-cyan to-meesho-purple"
          }].map((item, index) => <div key={index} className="text-center group">
                <div className={`w-24 h-24 bg-gradient-to-br ${item.color} rounded-3xl flex items-center justify-center mx-auto mb-6 text-white transform group-hover:scale-110 transition-all duration-500 shadow-lg group-hover:shadow-xl`}>
                  {item.icon}
                </div>
                <div className="bg-white rounded-2xl p-8 shadow-lg group-hover:shadow-xl transition-all duration-500 transform group-hover:-translate-y-2">
                  <div className="text-4xl font-bold text-meesho-purple mb-2">{item.number}</div>
                  <h3 className="font-heading text-xl font-semibold text-gray-800 mb-4">{item.title}</h3>
                  <p className="text-gray-600 leading-relaxed">{item.description}</p>
                </div>
              </div>)}
          </div>
        </div>
      </section>

      {/* Pain Points Breakdown */}
      <section className="py-24 bg-white">
        <div className="container mx-auto px-6">
          <div className="max-w-6xl mx-auto">
            <div className="grid lg:grid-cols-2 gap-16 items-center">
              <div>
                <h2 className="font-heading text-4xl font-bold text-meesho-purple mb-8">
                  Why Manual Tagging Fails at Scale
                </h2>
                <p className="text-xl text-gray-600 mb-12 leading-relaxed">
                  Traditional approaches to fashion attribute tagging face insurmountable challenges in today's fast-paced e-commerce environment.
                </p>
                
                <div className="space-y-6">
                  {[{
                  icon: <Target className="h-6 w-6" />,
                  title: "Inconsistency Crisis",
                  description: "Different taggers interpret the same garment differently, leading to conflicting attributes across similar products.",
                  impact: "40% accuracy variance"
                }, {
                  icon: <Clock className="h-6 w-6" />,
                  title: "Speed Bottleneck",
                  description: "Manual tagging creates massive delays, with products taking days to become searchable after upload.",
                  impact: "10x slower than needed"
                }, {
                  icon: <TrendingUp className="h-6 w-6" />,
                  title: "Scalability Nightmare",
                  description: "Growing product catalogs require exponentially more human resources, making scaling impossible.",
                  impact: "Linear cost growth"
                }].map((item, index) => <div key={index} className="flex items-start space-x-4 group">
                      <div className="w-12 h-12 bg-gradient-to-br from-meesho-purple to-meesho-magenta rounded-xl flex items-center justify-center text-white flex-shrink-0 group-hover:scale-110 transition-transform duration-300">
                        {item.icon}
                      </div>
                      <div className="flex-1">
                        <h3 className="font-heading text-xl font-semibold text-gray-800 mb-2">{item.title}</h3>
                        <p className="text-gray-600 mb-2">{item.description}</p>
                        <span className="inline-block px-3 py-1 bg-meesho-orange/10 text-meesho-orange text-sm font-semibold rounded-full">
                          {item.impact}
                        </span>
                      </div>
                    </div>)}
                </div>
              </div>

              <div className="relative">
                <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-3xl p-12 relative overflow-hidden">
                  <div className="absolute top-4 right-4 w-20 h-20 bg-meesho-orange/10 rounded-2xl rotate-12"></div>
                  <div className="absolute bottom-4 left-4 w-16 h-16 bg-meesho-purple/10 rounded-xl -rotate-6"></div>
                  
                  <div className="text-center">
                    <div className="text-6xl font-bold text-meesho-purple mb-4">83%</div>
                    <div className="text-xl font-semibold text-gray-800 mb-2">Resource Waste</div>
                    <p className="text-gray-600">
                      Of manual tagging efforts result in inconsistent or incomplete attribute data
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Business Impact Section */}
      <section className="py-24 bg-gradient-to-br from-gray-50 to-white">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="font-heading text-4xl md:text-5xl font-bold text-meesho-purple mb-6">
              Real Business Impact
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              How AI-powered fashion tagging transforms business outcomes across the entire e-commerce ecosystem
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
            {[{
            icon: <TrendingUp className="h-10 w-10" />,
            metric: "35%",
            title: "Search Improvement",
            description: "Increase in product discoverability and search relevance with accurate AI tagging",
            gradient: "from-meesho-purple to-meesho-magenta"
          }, {
            icon: <Zap className="h-10 w-10" />,
            metric: "90%",
            title: "Time Reduction",
            description: "From 10 minutes to 30 seconds per product with automated attribute prediction",
            gradient: "from-meesho-orange to-meesho-magenta"
          }, {
            icon: <CheckCircle className="h-10 w-10" />,
            metric: "95%",
            title: "Consistency Rate",
            description: "Uniform tagging standards maintained across entire product catalog",
            gradient: "from-meesho-cyan to-meesho-purple"
          }].map((item, index) => <Card key={index} className="bg-white shadow-lg hover:shadow-xl transition-all duration-500 transform hover:-translate-y-2 border-0 rounded-2xl overflow-hidden">
                <CardContent className="p-8 text-center">
                  <div className={`w-20 h-20 bg-gradient-to-br ${item.gradient} rounded-2xl flex items-center justify-center mx-auto mb-6 text-white shadow-lg`}>
                    {item.icon}
                  </div>
                  <div className="text-4xl font-bold text-meesho-purple mb-2">{item.metric}</div>
                  <h3 className="font-heading text-xl font-semibold text-gray-800 mb-4">{item.title}</h3>
                  <p className="text-gray-600 leading-relaxed">{item.description}</p>
                </CardContent>
              </Card>)}
          </div>
        </div>
      </section>

      {/* Solution Preview Section */}
      <section className="py-24 bg-white">
        <div className="container mx-auto px-6">
          <div className="max-w-4xl mx-auto text-center">
            <div className="mb-8">
              <span className="inline-flex items-center px-4 py-2 bg-meesho-purple/10 rounded-full text-meesho-purple text-sm font-medium mb-6">
                <Sparkles className="w-4 h-4 mr-2" />
                Our Solution
              </span>
            </div>
            
            <h2 className="font-heading text-4xl md:text-5xl font-bold text-meesho-purple mb-8">
              Deep Learning Architecture
            </h2>
            
            <p className="text-xl text-gray-600 mb-12 leading-relaxed">
              Our multi-label classification system combines transfer learning, attention mechanisms, and ensemble methods to achieve human-level accuracy at machine speed.
            </p>

            <div className="bg-gradient-to-r from-meesho-purple/5 to-meesho-orange/5 rounded-3xl p-12 mb-12">
              <div className="grid md:grid-cols-2 gap-12">
                <div className="text-left">
                  <h3 className="font-heading text-2xl font-semibold text-meesho-purple mb-4">Key Features</h3>
                  <div className="space-y-4">
                    {["Multi-label classification for complex attributes", "Transfer learning with fashion-specific fine-tuning", "Attention mechanisms for region-focused analysis", "Ensemble methods for improved robustness"].map((feature, index) => <div key={index} className="flex items-center">
                        <CheckCircle className="w-5 h-5 text-meesho-orange mr-3 flex-shrink-0" />
                        <span className="text-gray-700">{feature}</span>
                      </div>)}
                  </div>
                </div>
                
                <div className="text-left">
                  <h3 className="font-heading text-2xl font-semibold text-meesho-purple mb-4">Performance</h3>
                  <div className="space-y-4">
                    {[{
                    label: "Accuracy",
                    value: "94.7%"
                  }, {
                    label: "Processing Speed",
                    value: "< 30s"
                  }, {
                    label: "Attributes Detected",
                    value: "50+"
                  }, {
                    label: "Model Confidence",
                    value: "92%"
                  }].map((metric, index) => <div key={index} className="flex justify-between items-center">
                        <span className="text-gray-700">{metric.label}</span>
                        <span className="font-semibold text-meesho-purple">{metric.value}</span>
                      </div>)}
                  </div>
                </div>
              </div>
            </div>

            <Link to="/solution">
              <Button size="lg" className="bg-gradient-meesho hover:opacity-90 text-white text-xl px-12 py-6 rounded-full font-semibold shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-300">
                <Zap className="mr-3 h-6 w-6" />
                Experience the Solution
                <ArrowRight className="ml-3 h-6 w-6" />
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-white py-16 border-t border-gray-100">
        <div className="container mx-auto px-6">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="mb-8 md:mb-0 text-center md:text-left">
              <div className="font-heading font-bold text-3xl text-meesho-purple mb-4">
                Fashion AI Tagger
              </div>
              <p className="text-gray-600 text-lg">Built with ❤️ at Meesho AI Challenge</p>
            </div>
            
            <div className="flex space-x-8 items-center">
              
            </div>
          </div>
        </div>
      </footer>
    </div>;
};
export default About;