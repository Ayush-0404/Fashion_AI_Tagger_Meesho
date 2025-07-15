import { useState, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ArrowDown, Upload, FileText, Github, Sparkles, Zap, Target, Shirt } from "lucide-react";
import { Link } from 'react-router-dom';
const Index = () => {
  const [isVisible, setIsVisible] = useState(false);
  useEffect(() => {
    setIsVisible(true);
  }, []);
  const scrollToSection = (id: string) => {
    document.getElementById(id)?.scrollIntoView({
      behavior: 'smooth'
    });
  };
  return <div className="min-h-screen bg-white">
      {/* Navigation */}
      <nav className="fixed top-0 w-full bg-white/95 backdrop-blur-md z-50 border-b border-gray-100 shadow-sm">
        <div className="container mx-auto px-6 py-4 flex justify-between items-center">
          <a href="#hero" onClick={() => scrollToSection('hero')} className="font-heading font-bold text-2xl text-meesho-purple">
            Fashion AI Tagger
          </a>

          <div className="hidden md:flex space-x-8">
            <a href="#about" onClick={() => scrollToSection('about')} className="px-1 py-2 flex items-center text-gray-700 hover:text-meesho-purple transition-colors font-medium">About</a>
            <a href="#solution" onClick={() => scrollToSection('solution')} className="px-1 py-2 flex items-center text-gray-700 hover:text-meesho-purple transition-colors font-medium">Solution</a>
            <a href="#future" onClick={() => scrollToSection('future')} className="px-1 py-2 flex items-center text-gray-700 hover:text-meesho-purple transition-colors font-medium">Future</a>
            <Link to="/solution" className="bg-meesho-orange text-white px-6 py-2 rounded-full font-semibold hover:bg-orange-500 transition-all duration-300 transform hover:scale-105">
              Try Now
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section id="hero" className="relative min-h-screen flex items-center justify-center overflow-hidden bg-gradient-meesho-hero">
        {/* 3D Floating Elements */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-20 left-10 w-16 h-16 bg-white/20 rounded-lg floating-elements isometric-cube"></div>
          <div className="absolute top-32 right-20 w-12 h-12 bg-meesho-orange/30 rounded-xl floating-elements" style={{
          animationDelay: '1s'
        }}></div>
          <div className="absolute bottom-40 left-32 w-20 h-20 bg-meesho-cyan/20 rounded-2xl floating-elements" style={{
          animationDelay: '2s'
        }}></div>
          <div className="absolute top-1/2 right-10 w-14 h-14 bg-white/15 rounded-lg floating-elements" style={{
          animationDelay: '3s'
        }}></div>
          <div className="absolute bottom-20 right-1/3 w-10 h-10 bg-meesho-magenta/25 rounded-full floating-elements" style={{
          animationDelay: '0.5s'
        }}></div>
        </div>
        
        <div className={`relative z-10 text-center px-6 max-w-6xl mx-auto transition-all duration-1000 ${isVisible ? 'animate-fade-in' : 'opacity-0'}`}>
          <div className="mb-8">
            <span className="inline-flex items-center px-4 py-2 bg-white/20 backdrop-blur-sm rounded-full text-white text-sm font-medium mb-6">
              <Sparkles className="w-4 h-4 mr-2" />
              Meesho AI Data Challenge
            </span>
          </div>
          
          <h1 className="font-heading text-5xl md:text-7xl font-bold text-white mb-8 leading-tight">
            Automating Fashion Intelligence with
            <span className="block mt-2 text-gradient-meesho bg-clip-text text-transparent bg-gradient-to-r from-white via-meesho-orange to-white">
              Deep Learning
            </span>
          </h1>
          
          <p className="text-xl md:text-2xl text-white/90 mb-12 max-w-4xl mx-auto leading-relaxed">
            Upload → Predict → Organize. Experience AI-powered fashion tagging that transforms how we understand clothing attributes.
          </p>
          
          <div className="flex flex-col md:flex-row gap-6 justify-center items-center">
            <Link to="/solution">
              <Button size="lg" className="bg-white text-meesho-purple hover:bg-gray-100 text-xl px-10 py-6 rounded-full font-semibold transform hover:scale-105 transition-all duration-300 shadow-lg hover:shadow-xl">
                <Zap className="mr-3 h-5 w-5" />
                Try Our Project
              </Button>
            </Link>
            <Button variant="outline" size="lg" onClick={() => scrollToSection('about')} className="bg-white/10 backdrop-blur-sm border-2 border-white/30 text-white hover:bg-white hover:text-meesho-purple text-xl px-10 py-6 rounded-full font-semibold transform hover:scale-105 transition-all duration-300">
              Learn More
              <ArrowDown className="ml-3 h-5 w-5" />
            </Button>
          </div>
        </div>
      </section>

      {/* About Section */}
      <section id="about" className="py-24 bg-white">
        <div className="container mx-auto px-6">
          <div className="text-center mb-20">
            <h2 className="font-heading text-5xl md:text-6xl font-bold text-meesho-purple mb-8">
              The Fashion Tagging Challenge
            </h2>
            <p className="text-2xl text-gray-600 max-w-4xl mx-auto leading-relaxed">
              Manual attribute tagging is a massive bottleneck in fashion e-commerce. Here's why AI is the solution.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-10 max-w-7xl mx-auto">
            <Card className="bg-gradient-meesho-card border-0 shadow-meesho hover:shadow-meesho-hover transition-all duration-500 transform hover:-translate-y-3 rounded-2xl overflow-hidden">
              <CardContent className="p-10 text-center">
                <div className="w-20 h-20 bg-gradient-to-br from-meesho-magenta to-meesho-purple rounded-2xl flex items-center justify-center mx-auto mb-8 shadow-lg">
                  <span className="text-white text-3xl font-bold">1M+</span>
                </div>
                <h3 className="font-heading text-2xl font-semibold text-meesho-purple mb-6">Scale Problem</h3>
                <p className="text-gray-600 text-lg leading-relaxed">Millions of fashion SKUs need accurate attribute tagging for searchability and categorization.</p>
              </CardContent>
            </Card>

            <Card className="bg-gradient-meesho-card border-0 shadow-meesho hover:shadow-meesho-hover transition-all duration-500 transform hover:-translate-y-3 rounded-2xl overflow-hidden">
              <CardContent className="p-10 text-center">
                <div className="w-20 h-20 bg-gradient-to-br from-meesho-orange to-meesho-magenta rounded-2xl flex items-center justify-center mx-auto mb-8 shadow-lg">
                  <span className="text-white text-3xl font-bold">80%</span>
                </div>
                <h3 className="font-heading text-2xl font-semibold text-meesho-purple mb-6">Manual Effort</h3>
                <p className="text-gray-600 text-lg leading-relaxed">Traditional tagging requires massive human resources and is prone to inconsistencies.</p>
              </CardContent>
            </Card>

            <Card className="bg-gradient-meesho-card border-0 shadow-meesho hover:shadow-meesho-hover transition-all duration-500 transform hover:-translate-y-3 rounded-2xl overflow-hidden">
              <CardContent className="p-10 text-center">
                <div className="w-20 h-20 bg-gradient-to-br from-meesho-purple to-meesho-darkPurple rounded-2xl flex items-center justify-center mx-auto mb-8 shadow-lg">
                  <Sparkles className="h-10 w-10 text-white" />
                </div>
                <h3 className="font-heading text-2xl font-semibold text-meesho-purple mb-6">Smart Solution</h3>
                <p className="text-gray-600 text-lg leading-relaxed">Deep learning models can automate this process with high accuracy and consistency.</p>
              </CardContent>
            </Card>
          </div>

          <div className="text-center mt-16">
            <Link to="/about">
              <Button variant="outline" size="lg" className="border-2 border-meesho-purple text-meesho-purple hover:bg-meesho-purple hover:text-white text-lg px-8 py-4 rounded-full font-semibold transition-all duration-300">
                Read Full Problem Statement
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Solution Section */}
      <section id="solution" className="py-24 bg-gradient-to-br from-gray-50 to-white">
        <div className="container mx-auto px-6">
          <div className="text-center mb-20">
            <h2 className="font-heading text-5xl md:text-6xl font-bold text-meesho-purple mb-8">
              How Our Solution Works
            </h2>
            <p className="text-2xl text-gray-600 max-w-4xl mx-auto leading-relaxed">
              A seamless pipeline from image upload to intelligent attribute prediction
            </p>
          </div>

          <div className="grid md:grid-cols-4 gap-8 max-w-8xl mx-auto">
            {[{
            step: "1",
            title: "Image Upload",
            description: "Upload fashion product images through our intuitive interface",
            icon: <Upload className="h-10 w-10" />,
            gradient: "from-meesho-magenta to-meesho-purple"
          }, {
            step: "2",
            title: "Preprocessing",
            description: "Images are automatically resized, normalized and prepared for inference",
            icon: <Target className="h-10 w-10" />,
            gradient: "from-meesho-orange to-meesho-magenta"
          }, {
            step: "3",
            title: "Deep Learning",
            description: "Our trained CNN model analyzes visual features and patterns",
            icon: <Sparkles className="h-10 w-10" />,
            gradient: "from-meesho-purple to-meesho-darkPurple"
          }, {
            step: "4",
            title: "Attributes",
            description: "Get comprehensive attribute predictions with confidence scores",
            icon: <FileText className="h-10 w-10" />,
            gradient: "from-meesho-cyan to-meesho-purple"
          }].map((item, index) => <div key={index} className="text-center group">
                <div className={`w-24 h-24 bg-gradient-to-br ${item.gradient} rounded-3xl flex items-center justify-center mx-auto mb-8 text-white transform group-hover:scale-110 group-hover:rotate-3 transition-all duration-500 shadow-lg group-hover:shadow-xl`}>
                  {item.icon}
                </div>
                <div className="bg-white rounded-2xl p-8 shadow-meesho group-hover:shadow-meesho-hover transition-all duration-500 transform group-hover:-translate-y-2">
                  <div className="text-sm font-bold text-meesho-orange mb-3 uppercase tracking-wider">Step {item.step}</div>
                  <h3 className="font-heading text-2xl font-semibold text-meesho-purple mb-4">{item.title}</h3>
                  <p className="text-gray-600 leading-relaxed">{item.description}</p>
                </div>
              </div>)}
          </div>

          <div className="text-center mt-16">
            <Link to="/solution">
              <Button size="lg" className="bg-gradient-meesho hover:opacity-90 text-white text-xl px-12 py-6 rounded-full font-semibold shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-300">
                <Zap className="mr-3 h-6 w-6" />
                Try Our Project
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* What's Next Section */}
      <section id="future" className="bg-gradient-meesho-hero text-white relative overflow-hidden py-[50px]">
        {/* Clothing Graphics Background */}
        <div className="absolute inset-0 pointer-events-none opacity-10">
          <div className="absolute top-20 left-10 transform rotate-12">
            <Shirt className="w-16 h-16 text-white floating-elements" />
          </div>
          <div className="absolute top-32 right-20 transform -rotate-12" style={{
          animationDelay: '1s'
        }}>
            <Shirt className="w-12 h-12 text-meesho-orange floating-elements" />
          </div>
          <div className="absolute bottom-40 left-32 transform rotate-45" style={{
          animationDelay: '2s'
        }}>
            <Shirt className="w-20 h-20 text-meesho-cyan floating-elements" />
          </div>
          <div className="absolute top-1/2 right-10 transform -rotate-6" style={{
          animationDelay: '3s'
        }}>
            <Shirt className="w-14 h-14 text-white floating-elements" />
          </div>
          <div className="absolute bottom-20 right-1/3 transform rotate-12" style={{
          animationDelay: '0.5s'
        }}>
            <Shirt className="w-10 h-10 text-meesho-magenta floating-elements" />
          </div>
          
          {/* Additional decorative elements */}
          <div className="absolute top-1/4 left-1/4 w-3 h-3 bg-white/30 rounded-full floating-elements"></div>
          <div className="absolute bottom-1/3 right-1/4 w-2 h-2 bg-meesho-orange/40 rounded-full floating-elements" style={{
          animationDelay: '1.5s'
        }}></div>
          <div className="absolute top-3/4 left-1/3 w-4 h-4 bg-meesho-cyan/30 rounded-full floating-elements" style={{
          animationDelay: '2.5s'
        }}></div>
        </div>
        
        <div className="container mx-auto px-6 relative z-10">
          <div className="text-center">
            <div className="mb-8">
              <span className="inline-flex items-center px-4 py-2 bg-white/20 backdrop-blur-sm rounded-full text-white text-sm font-medium mb-6">
                <Sparkles className="w-4 h-4 mr-2" />
                Future Roadmap
              </span>
            </div>
            
            <h2 className="font-heading text-4xl md:text-5xl font-bold mb-6">
              What's Next?
            </h2>
            <p className="text-xl md:text-2xl text-white/90 max-w-4xl mx-auto leading-relaxed mb-8">
              We will be exploring more real-world problems using Machine Learning and development to create innovative solutions for the digital world.
            </p>
            
            {/* Fashion Icons Grid */}
            <div className="flex justify-center items-center space-x-8">
              <div className="w-16 h-16 bg-white/20 backdrop-blur-sm rounded-2xl flex items-center justify-center transform hover:scale-110 transition-transform duration-300">
                <Shirt className="w-8 h-8 text-white" />
              </div>
              <div className="w-20 h-20 bg-meesho-orange/30 backdrop-blur-sm rounded-2xl flex items-center justify-center transform hover:scale-110 transition-transform duration-300">
                <Zap className="w-10 h-10 text-white" />
              </div>
              <div className="w-16 h-16 bg-white/20 backdrop-blur-sm rounded-2xl flex items-center justify-center transform hover:scale-110 transition-transform duration-300">
                <Target className="w-8 h-8 text-white" />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer Section */}
      <section className="bg-white py-16 border-t border-gray-100">
        <div className="container mx-auto px-6">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="mb-8 md:mb-0 text-center md:text-left">
              <div className="font-heading font-bold text-3xl text-meesho-purple mb-4">
                Fashion AI Tagger
              </div>
              <p className="text-gray-600 text-lg">Built with ❤️ at Meesho AI Challenge</p>
            </div>
            
            <div className="flex space-x-8 items-center">
              <a href="#" className="text-gray-600 hover:text-meesho-purple transition-colors p-2 transform hover:scale-110 duration-300">
                <Github className="h-8 w-8" />
              </a>
              <a href="mailto:contact@fashionai.com" className="bg-meesho-purple hover:bg-meesho-darkPurple text-white px-6 py-3 rounded-full font-semibold transition-all duration-300 transform hover:scale-105">
                Contact Us
              </a>
            </div>
          </div>
        </div>
      </section>
    </div>;
};
export default Index;