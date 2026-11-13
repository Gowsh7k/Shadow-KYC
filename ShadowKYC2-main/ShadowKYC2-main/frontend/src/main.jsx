import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'
import { clearAuth, getToken } from './lib/auth'

// Global 401 handler: if the backend rejects a token (expired/revoked),
// clear local auth state and send the user to login.
const originalFetch = window.fetch;
window.fetch = async (input, init) => {
    const res = await originalFetch(input, init);
    if (res.status === 401) {
        const url = String(input);
        // Avoid redirect loops on login/auth endpoints themselves.
        if (!url.includes('/auth/login') && !url.includes('/auth/signup')) {
            clearAuth();
            if (window.location.pathname !== '/login') {
                window.location.href = '/login';
            }
        }
    }
    return res;
};

// Ensure outgoing requests carry the stored token when one is present.
const originalOpen = XMLHttpRequest.prototype.open;
XMLHttpRequest.prototype.open = function (method, url, ...args) {
    if (typeof url === 'string' && !String(url).includes('/auth/login') && !String(url).includes('/auth/signup')) {
        const token = getToken();
        if (token) {
            this.setRequestHeader('Authorization', 'Bearer ' + token);
        }
    }
    return originalOpen.call(this, method, url, ...args);
};

ReactDOM.createRoot(document.getElementById('root')).render(
    <App />
)
