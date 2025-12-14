import bg_image from './assets/bg1.png'
import logo from './assets/logo1.png'
import element1 from './assets/element1.png'
import element2 from './assets/element2.png'
import element3 from './assets/element3.png'
import element4 from './assets/element4.png'
import element5 from './assets/element5.png'
import element6 from './assets/element6.png'
import num from './assets/1234.png'
import demovideo from './assets/demo-video.mp4'
import './App.css'
import Uploader from './pages/uploader_page' 
import LoginForm from './pages/login'   

import { useNavigate } from "react-router-dom";

function HomePage() {
    const navigate = useNavigate();
    return (
        <div>
            <div className="relative top-0 left-0  overflow-hidden w-full">
                <img src={bg_image} alt="Background" className="fixed inset-0 object-cover w-full -z-10" />
                <nav className="flex  items-center justify-center backdrop-blur-sm mt-0 ">
                    <img src={logo} alt="Logo" className="mb-8 w-20 h-20" />
                    <h1 className="text-2xl font-bold text-white mb-4 mr-60">Demand Cast</h1>
                    <ul className="hidden md:flex gap-6 ml-50 ">
                        <li><a href="#home" className="hover:text-red-600 -2xl">Home</a></li>
                        <li><a href="#about" className="hover:text-red-600">About us</a></li>
                        <li><a href="#services" className="hover:text-red-600">Services</a></li>
                        <li><a href="#contact" className="hover:text-red-600">Contact</a></li>
                    </ul>
                    <button className="bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700 transition " onClick={() => navigate("/login")}>login</button>
                </nav>
                <div className='flex flex-row justify-center w=1/2 -mt-10'>
                    <div className="flex flex-col items-center justify-center min-h-screen text-center px-4">
                        <h1 className="text-5xl font-bold text-white mb-4 text-left">Unlock Peak Performance with Untelligence Analytics</h1>
                        <h3 className="text-2xl text-white mb-8 text-left">Get universal insights on demand and supply chain with better sales and inventory predictions</h3>
                        <div className='flex flex-row items-start justify-center '
                        >
                            <button className="bg-red-600 text-white px-6 py-3 rounded hover:bg-red-700 transition mr-4">request demo</button>
                            <button className='backdrop-blur-2xl rounded-2xl' onClick={() => navigate("/next")}>Learn More</button>
                        </div>

                    </div>
                    <div className='flex flex-col items-center justify-center min-h-screen text-center px-4 w-1/2'>
                    </div>
                </div>
            </div>
            <div className='text-xl text-white -mt-20 mb-20'>
                <h1 className='text-xl text-white font-serif font-extrabold'>What are you waiting for ?</h1>
                <button className='text-xl rounded-2xl border font-black' onClick={() => navigate("/next")} >lets get started</button>
            </div>
            <div className=" backdrop-blur-2xl flex flex-col items-center justify-center min-h-screen text-center px-4">
                <h2 className="text-4xl font-bold text-white mb-8">
                    The Challenges & Our Solutions
                </h2>

                {/* Container for half-half layout */}
                <div className="flex flex-col md:flex-row items-stretch justify-center gap-8 w-full max-w-6xl">

                    {/* Left Side — Challenges */}
                    <div className="  flex flex-col justify-top items-center p-8 rounded-2xl w-full md:w-1/2 shadow-lg transition hover:scale-105">
                        <h3 className="text-3xl font-bold text-white mb-6 underline decoration-red-500 text-left font-serif">
                            Challenges
                        </h3>
                        <div className="flex flex-row justify-center items-center">
                            <div className="flex flex-col text-left">
                                <p className="text-2xl font-bold text-black mb-5"> *Missed Opportunities</p>
                                <p className="text-2xl font-bold text-black mb-5">*Inventory Imbalance</p>
                                <p className="text-2xl font-bold text-black mb-5">*Inefficient Operations</p>
                            </div>
                            <img src={element1} alt="Challenges Icon" className="w-60 h-60" />
                        </div>
                    </div>

                    {/* Right Side — Solutions */}
                    <div className="  flex flex-col justify-top items-center p-8 rounded-2xl w-full md:w-1/2 shadow-lg transition hover:scale-105">
                        <h3 className="text-3xl font-bold text-white mb-6 underline decoration-red-500">
                            Our Solutions
                        </h3>

                        <div className="flex flex-col gap-6">
                            <div className="flex flex-row items-center justify-start">
                                <img src={element2} alt="Element 2" className="w-24 h-24 mr-4" />
                                <p className="text-2xl font-bold text-black">
                                    Supply Chain Optimization
                                </p>
                            </div>
                            <div className="flex flex-row items-center justify-start">
                                <img src={element3} alt="Element 3" className="w-24 h-24 mr-4" />
                                <p className="text-2xl font-bold text-black">
                                    Business Performance Metrics
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
                <div className='flex flex-row items-center   mt-20'>
                    <div className='flex flex-col justify-center items-center w-1/2'>
                        <h1 className='text-left font-extrabold to-black text-4xl mt-20 mb-10 '>Key Analytics Pillars</h1>
                        <img src={num} alt="Key Analytics Pillars" className="w-full h-auto " />
                        <div className='flex flex-row items-center justify-between gap-13 mr-5 text-2xl text-amber-50 font-medium'>
                        <p className='font-xl font-medium text-amber-50'>Demand </p>
                        <p className='font-xl font-medium text-amber-50'> Supply </p>
                        <p className='font-xl font-medium text-amber-50'> Inventory </p>
                        <p className='font-xl font-medium text-amber-50'> sales</p>
                        </div>
                    </div>
                    <div className='flex flex-row items-center  justify-center w-1/2 gap-2 '>
                        <div className='flex flex-col'>

                            <img src={element4} alt="Element 4" className=" w-60 mb-10 mt-10" />
                            <p className='text-amber-50 text-2xl'>AI Powered Algorithms</p>
                        </div>
                        <div className='flex flex-col'
                        >
                            <img src={element5} alt="Element 5" className="w-60 mb-10 mt-10" />
                            <p className='text-amber-50 text-2xl'>Customizale Dashboards</p>
                        </div>
                        <div className='flex flex-col'>
                            <img src={element6} alt="Element 6" className="w-60  mb-10 mt-10" />
                            <p className='text-amber-50 text-2xl'>Seamless Integration</p>
                        </div>
                    </div>
                </div>
                <div className='flex flex-row'>
                    <div className='flex flex-col justify-center items-center w-1/2'>
                        <h2 className="text-5xl font-bold text-white -mb-4 mt-20">how it works?</h2>
                        <video src={demovideo} alt='controls' controls className='h-100 w-auto' />
                    </div>
                    <div className='flex flex-col justify-center items-center w-1/2'>
                        <h1 className='text-4xl font-bold text-white mt-32 '>Watch Our Demo Video to See Demand Cast in Action!</h1>
                        <h2 className="text-4xl font-bold text-white mb-8 mt-10">Get Started with Demand Cast Today!</h2>
                    </div>
                </div>
                        <h1 className='font-black text-4xl'>success stories </h1>
            </div>
            <footer className=" text-white py-6 mt-40">
                <div className='flex flex-row items-center justify-center'>
                    <p className='text-4xl text-amber-50  font-serif font-extrabold'>Ready to Transform Your Business?</p>
                    {/* <button className="bg-red-600 text-white px-6 py-3 rounded hover:bg-red-700 transition ml-6">Contact Us</button> */}
                    <p className='text-2xl text-amber-50  font-serif font-extrabold ml-6'>Stay Informed!</p>
                    <input type="email" placeholder="Enter your email" className="ml-4 p-2 rounded  text-white text-2xl border" />
                    <button className="bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700 transition ml-4">Subscribe</button>
                </div>
                <div className="backdrop-blur-2xl container mx-auto px-4 text-center mt-10">
                    <p>&copy; Love my work? ☕ Buy me a coffee and keep the code flowing!</p>
                    <p>&copy; 2024 Demand Cast. All rights reserved.</p>
                </div>
            </footer>
        </div>
    )
}

export default HomePage