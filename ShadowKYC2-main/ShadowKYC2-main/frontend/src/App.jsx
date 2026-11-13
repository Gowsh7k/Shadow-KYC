import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Login from './pages/Login';
import Signup from './pages/Signup';
import LandingPage from './pages/LandingPage';
import Home from './pages/Home';
import DevDashboard from './pages/DevDashboard';
import TenantMonitor from './pages/TenantMonitor';
import UserCamera from './pages/UserCamera';
import DocumentUpload from './pages/DocumentUpload';
import { ThemeProvider } from './ThemeContext';
import { isAuthenticated } from './lib/auth';
import Footer from './components/Footer';

const ProtectedRoute = ({ children }) => {
    if (!isAuthenticated()) {
        return <Navigate to="/login" replace />;
    }
    return children;
};

const App = () => {
    return (
        <ThemeProvider>
            <BrowserRouter>
                <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', background: 'var(--bg-color)', color: 'var(--text-main)', transition: 'background 0.3s, color 0.3s' }}>
                    <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                        <Routes>
                            {/* Public Routes */}
                            <Route path="/" element={<LandingPage />} />
                            <Route path="/login" element={<Login />} />
                            <Route path="/signup" element={<Signup />} />

                            {/* Core Application Routes */}
                            <Route path="/home" element={<ProtectedRoute><Home /></ProtectedRoute>} />
                            <Route path="/dev" element={<ProtectedRoute><DevDashboard /></ProtectedRoute>} />

                            {/* KYC Verification Sessions */}
                            <Route path="/tenant/:sessionId" element={<ProtectedRoute><TenantMonitor /></ProtectedRoute>} />
                            <Route path="/verify/:sessionId" element={<ProtectedRoute><UserCamera /></ProtectedRoute>} />
                            <Route path="/client/:sessionId" element={<ProtectedRoute><UserCamera /></ProtectedRoute>} />
                            <Route path="/upload/:sessionId" element={<ProtectedRoute><DocumentUpload /></ProtectedRoute>} />

                            {/* Catch-all */}
                            <Route path="*" element={<Navigate to="/" replace />} />
                        </Routes>
                    </div>
                    <Footer />
                </div>
            </BrowserRouter>
        </ThemeProvider>
    );
};

export default App;
