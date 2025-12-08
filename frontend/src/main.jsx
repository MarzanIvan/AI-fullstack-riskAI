import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import './index.css'
import './main.min.css'

import App from './App.jsx'
import PageNotFound from './PageNotFound.jsx'
import Contacts from './Contacts.jsx'
import NotFound from './NotFound.jsx'
import Projects from './Projects.jsx'
import Credit from './credit.jsx'
import Investment from './investment.jsx'
import Insurance from "./insurance.jsx";

createRoot(document.getElementById('root')).render(
    <StrictMode>
        <BrowserRouter>
            <Routes>
                <Route index element={<App />} />
                <Route path="/projects" element={<Projects />} />
                <Route path="/contacts" element={<Contacts />} />
                <Route path="/credit" element={<Credit />} />
                <Route path="/investment" element={<Investment />} />
                <Route path="/insurance" element={<Insurance />} />
                <Route path="*" element={<NotFound />} />
            </Routes>
        </BrowserRouter>
    </StrictMode>
)