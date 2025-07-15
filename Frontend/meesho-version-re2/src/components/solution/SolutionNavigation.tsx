import { Link } from 'react-router-dom';
const SolutionNavigation = () => {
  return <nav className="bg-white/90 backdrop-blur-md border-b border-purple-100 sticky top-0 z-50">
      <div className="container mx-auto px-6 py-4 flex justify-between items-center">
        <Link to="/" className="font-bold text-xl bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
          Fashion AI Tagger
        </Link>
        <div className="flex space-x-6">
          
          <Link to="/about" className="text-gray-700 hover:text-purple-600 transition-colors font-medium">About</Link>
        </div>
      </div>
    </nav>;
};
export default SolutionNavigation;