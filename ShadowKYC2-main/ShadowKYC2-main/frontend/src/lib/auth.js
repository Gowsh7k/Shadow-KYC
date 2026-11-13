/**
 * frontend/src/lib/auth.js
 * ─────────────────────────
 * Client-side auth helpers for the LiveGuard bank-oriented backend.
 * The backend remains the source of truth for roles/authorization; the
 * frontend never trusts a role supplied by localStorage alone.
 */

const TOKEN_KEY = "shadow_token";
const ROLE_KEY = "shadow_role";
const USER_KEY = "shadow_user";
const API_BASE = ""; // same-origin (Vite proxy forwards /auth, /kyc, /docs, /session, /uploads)

export function getToken() {
    return localStorage.getItem(TOKEN_KEY) || null;
}

export function getRole() {
    return localStorage.getItem(ROLE_KEY) || null;
}

export function getUser() {
    try {
        const raw = localStorage.getItem(USER_KEY);
        return raw ? JSON.parse(raw) : null;
    } catch (e) {
        return null;
    }
}

/** Decode a JWT payload without verification (client-side expiry check only). */
export function decodeToken(token) {
    if (!token) return null;
    try {
        const parts = token.split(".");
        if (parts.length !== 3) return null;
        return JSON.parse(atob(parts[1]));
    } catch (e) {
        return null;
    }
}

/** True if a token is present and not yet expired. Expired tokens are cleared. */
export function isAuthenticated() {
    const token = getToken();
    if (!token) return false;
    const payload = decodeToken(token);
    if (!payload || !payload.exp) return false;
    const now = Math.floor(Date.now() / 1000);
    if (now >= payload.exp) {
        clearAuth();
        return false;
    }
    return true;
}

export function setAuth(token, user, role) {
    localStorage.setItem(TOKEN_KEY, token);
    localStorage.setItem(ROLE_KEY, role || (user && user.role) || "user");
    localStorage.setItem(USER_KEY, JSON.stringify(user || {}));
}

export function clearAuth() {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(ROLE_KEY);
    localStorage.removeItem(USER_KEY);
}

export function authHeader() {
    const token = getToken();
    return token ? { Authorization: `Bearer ${token}` } : {};
}

export function isClient() {
    return getRole() === "CLIENT";
}

export function isOfficer() {
    const r = getRole();
    return r === "BANK_OFFICER" || r === "BANK_ADMIN" || r === "PLATFORM_ADMIN";
}

export function isOfficerOrAbove() {
    return isOfficer();
}

/** API helper: JSON request with auth header. Throws on non-ok responses. */
export async function apiRequest(path, options = {}) {
    const res = await fetch(`${API_BASE}${path}`, {
        ...options,
        headers: { ...authHeader(), ...(options.headers || {}) },
    });
    if (!res.ok) {
        const errText = await res.text().catch(() => "");
        const err = new Error(errText || `Request failed (${res.status})`);
        err.status = res.status;
        throw err;
    }
    const ct = res.headers.get("content-type") || "";
    if (ct.includes("application/json")) return res.json();
    return res.text();
}
