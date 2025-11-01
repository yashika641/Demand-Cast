import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import HomePage from "./home_page";
import Uploader from "./pages/uploader_page";
import Services from "./pages/services";
export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/next" element={<Uploader />} />
        <Route path="/services" element={<Services />} />
      </Routes>
    </Router>
  );
}
