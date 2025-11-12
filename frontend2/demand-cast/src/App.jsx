import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import HomePage from "./home_page";
import Uploader from "./pages/uploader_page";
import Services from "./pages/services";
import Sales from "./pages/sales";
import LoginForm from "./pages/login";
import SignupForm from "./pages/signup";
import ProductDashboard from "./pages/product_catalogue"; 
import CustomerDashboard from "./pages/customer";

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/Uploader" element={<Uploader />} />
        <Route path="/services" element={<Services />} />
        <Route path="/services/sales-analytics" element={<Sales />} />
        <Route path="/login" element={<LoginForm />} />
        <Route path="/signup" element={<SignupForm />} />
        <Route path="/services/product-lifecycle" element={<ProductDashboard />} />
        <Route path="/services/customer-analytics" element={<CustomerDashboard />} />
      </Routes>
    </Router>
  );
}
