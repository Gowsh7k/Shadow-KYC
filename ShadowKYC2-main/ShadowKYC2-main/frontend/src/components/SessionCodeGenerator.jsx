import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
    Shield, Plus, Copy, Check, QrCode, Monitor, Clock, 
    RefreshCw, AlertTriangle, ExternalLink, Sparkles, 
    User, Hash, ArrowRight, X, ShieldCheck, CheckCircle2,
    Calendar, Share2, Layers
} from 'lucide-react';

const SessionCodeGenerator = ({ onSessionCreated }) => {
    const navigate = useNavigate();
    const [generating, setGenerating] = useState(false);
    const [error, setError] = useState('');
    const [currentCode, setCurrentCode] = useState('');
    const [customerRef, setCustomerRef] = useState('');
    const [expiryMinutes, setExpiryMinutes] = useState(30);
    const [profile, setProfile] = useState('full');
    const [copiedCode, setCopiedCode] = useState(false);
    const [copiedLink, setCopiedLink] = useState(false);
    const [showQR, setShowQR] = useState(false);
    const [timeLeft, setTimeLeft] = useState(null);
    const [recentSessions, setRecentSessions] = useState(() => {
        try {
            const saved = localStorage.getItem('shadow_officer_sessions');
            return saved ? JSON.parse(saved) : [];
        } catch (e) {
            return [];
        }
    });

    // Save recent sessions
    useEffect(() => {
        try {
            localStorage.setItem('shadow_officer_sessions', JSON.stringify(recentSessions));
        } catch (e) { }
    }, [recentSessions]);

    // Expiry countdown timer
    useEffect(() => {
        if (!currentCode || timeLeft === null) return;
        if (timeLeft <= 0) return;

        const interval = setInterval(() => {
            setTimeLeft(prev => {
                if (prev <= 1) {
                    clearInterval(interval);
                    return 0;
                }
                return prev - 1;
            });
        }, 1000);

        return () => clearInterval(interval);
    }, [currentCode, timeLeft]);

    const formatCountdown = (seconds) => {
        if (seconds === null) return '';
        const m = Math.floor(seconds / 60);
        const s = seconds % 60;
        return `${m}:${s < 10 ? '0' : ''}${s}`;
    };

    const clientInviteUrl = currentCode 
        ? `${window.location.origin}/verify/${currentCode}`
        : '';

    const handleGenerate = async () => {
        setGenerating(true);
        setError('');
        setCopiedCode(false);
        setCopiedLink(false);

        try {
            const token = localStorage.getItem('shadow_token') || '';
            if (!token) {
                setError('Authentication token missing. Please log in again.');
                setGenerating(false);
                return;
            }

            const res = await fetch('/session/create', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                }
            });

            if (res.status === 401) {
                setError('Session expired or unauthorized. Please re-authenticate.');
                setGenerating(false);
                return;
            }

            const data = await res.json();
            if (!res.ok) {
                setError(data.detail || 'Failed to generate session code.');
                setGenerating(false);
                return;
            }

            const code = data.session_id;
            setCurrentCode(code);
            setTimeLeft(expiryMinutes * 60);

            // Add to recent sessions list
            const newRecord = {
                code,
                customerRef: customerRef.trim() || 'General Customer',
                profile,
                expiryMinutes,
                createdAt: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                timestamp: Date.now()
            };
            setRecentSessions(prev => [newRecord, ...prev.filter(s => s.code !== code)].slice(0, 10));

            if (onSessionCreated) {
                onSessionCreated(code);
            }
        } catch (err) {
            console.error('Session generation error:', err);
            setError('Network error: Unable to reach the server. Is backend running?');
        } finally {
            setGenerating(false);
        }
    };

    const copyCode = (codeToCopy = currentCode) => {
        if (!codeToCopy) return;
        navigator.clipboard.writeText(codeToCopy);
        setCopiedCode(true);
        setTimeout(() => setCopiedCode(false), 2000);
    };

    const copyLink = (linkToCopy = clientInviteUrl) => {
        if (!linkToCopy) return;
        navigator.clipboard.writeText(linkToCopy);
        setCopiedLink(true);
        setTimeout(() => setCopiedLink(false), 2000);
    };

    const removeSession = (codeToRemove) => {
        setRecentSessions(prev => prev.filter(s => s.code !== codeToRemove));
        if (currentCode === codeToRemove) {
            setCurrentCode('');
        }
    };

    return (
        <div className="glass-card" style={{
            padding: '2.25rem',
            borderRadius: '1.25rem',
            border: '1px solid var(--accent)',
            background: 'var(--glass-bg)',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.12)',
            position: 'relative',
            overflow: 'hidden',
            marginBottom: '2rem'
        }}>
            {/* Top decorative gradient bar */}
            <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                height: '4px',
                background: 'linear-gradient(90deg, #FFC107, #FF9800, #F59E0B)'
            }} />

            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <div style={{
                        width: '52px',
                        height: '52px',
                        borderRadius: '1rem',
                        background: 'rgba(255, 193, 7, 0.15)',
                        border: '1px solid rgba(255, 193, 7, 0.3)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        boxShadow: '0 0 20px rgba(255, 193, 7, 0.2)'
                    }}>
                        <Hash size={26} color="#FFC107" />
                    </div>
                    <div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                            <h2 style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--text-main)', margin: 0 }}>
                                Session Code Generator
                            </h2>
                            <span style={{
                                fontSize: '0.7rem',
                                fontWeight: 700,
                                padding: '0.2rem 0.6rem',
                                borderRadius: '2rem',
                                background: 'rgba(255, 193, 7, 0.18)',
                                color: '#FFC107',
                                textTransform: 'uppercase',
                                letterSpacing: '0.05em'
                            }}>
                                Officer Portal
                            </span>
                        </div>
                        <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                            Generate an ephemeral 6-digit cryptographic pairing code for customer Video KYC verification.
                        </p>
                    </div>
                </div>

                {currentCode && (
                    <button
                        onClick={() => { setCurrentCode(''); setError(''); }}
                        style={{
                            background: 'rgba(255, 255, 255, 0.05)',
                            border: '1px solid var(--glass-border)',
                            borderRadius: '0.6rem',
                            padding: '0.45rem 0.85rem',
                            color: 'var(--text-secondary)',
                            fontSize: '0.85rem',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.4rem'
                        }}
                    >
                        <RefreshCw size={14} /> New Code
                    </button>
                )}
            </div>

            {/* Error Banner */}
            {error && (
                <div style={{
                    marginBottom: '1.5rem',
                    padding: '0.85rem 1.25rem',
                    borderRadius: '0.75rem',
                    background: 'rgba(239, 68, 68, 0.1)',
                    border: '1px solid rgba(239, 68, 68, 0.25)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.75rem',
                    fontSize: '0.9rem',
                    color: '#ef4444'
                }}>
                    <AlertTriangle size={18} />
                    <span>{error}</span>
                </div>
            )}

            {/* Generator Form Controls (If no active code or to change parameters) */}
            {!currentCode ? (
                <div>
                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
                        gap: '1.25rem',
                        marginBottom: '1.5rem'
                    }}>
                        {/* Customer Tag / Reference */}
                        <div>
                            <label style={{ display: 'block', fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '0.4rem' }}>
                                <User size={14} style={{ display: 'inline', marginRight: '4px', verticalAlign: '-2px' }} />
                                Customer Name / Application Ref (Optional)
                            </label>
                            <input
                                type="text"
                                placeholder="e.g. Alice Smith (KYC-2041)"
                                value={customerRef}
                                onChange={e => setCustomerRef(e.target.value)}
                                style={{
                                    width: '100%',
                                    padding: '0.75rem 1rem',
                                    borderRadius: '0.75rem',
                                    background: 'var(--input-bg)',
                                    border: '1px solid var(--nav-border)',
                                    color: 'var(--text-main)',
                                    fontSize: '0.95rem',
                                    outline: 'none',
                                    boxSizing: 'border-box'
                                }}
                            />
                        </div>

                        {/* Validity Duration */}
                        <div>
                            <label style={{ display: 'block', fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '0.4rem' }}>
                                <Clock size={14} style={{ display: 'inline', marginRight: '4px', verticalAlign: '-2px' }} />
                                Session Code Validity
                            </label>
                            <select
                                value={expiryMinutes}
                                onChange={e => setExpiryMinutes(Number(e.target.value))}
                                style={{
                                    width: '100%',
                                    padding: '0.75rem 1rem',
                                    borderRadius: '0.75rem',
                                    background: 'var(--input-bg)',
                                    border: '1px solid var(--nav-border)',
                                    color: 'var(--text-main)',
                                    fontSize: '0.95rem',
                                    outline: 'none',
                                    cursor: 'pointer',
                                    boxSizing: 'border-box'
                                }}
                            >
                                <option value={15}>15 Minutes (Express / In-Person)</option>
                                <option value={30}>30 Minutes (Standard Verification)</option>
                                <option value={60}>1 Hour (Remote Scheduled Call)</option>
                                <option value={1440}>24 Hours (Asynchronous Self-Serve)</option>
                            </select>
                        </div>

                        {/* Security Profile */}
                        <div>
                            <label style={{ display: 'block', fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '0.4rem' }}>
                                <Layers size={14} style={{ display: 'inline', marginRight: '4px', verticalAlign: '-2px' }} />
                                Verification Profile
                            </label>
                            <select
                                value={profile}
                                onChange={e => setProfile(e.target.value)}
                                style={{
                                    width: '100%',
                                    padding: '0.75rem 1rem',
                                    borderRadius: '0.75rem',
                                    background: 'var(--input-bg)',
                                    border: '1px solid var(--nav-border)',
                                    color: 'var(--text-main)',
                                    fontSize: '0.95rem',
                                    outline: 'none',
                                    cursor: 'pointer',
                                    boxSizing: 'border-box'
                                }}
                            >
                                <option value="full">🛡️ Full 7-Layer Detection (AI + Anti-Spoof)</option>
                                <option value="liveness">⚡ Rapid Liveness & Blink Analysis</option>
                                <option value="docs">📄 Document OCR + Face Comparison</option>
                            </select>
                        </div>
                    </div>

                    {/* Action Button */}
                    <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                        <button
                            onClick={handleGenerate}
                            disabled={generating}
                            style={{
                                padding: '0.9rem 2rem',
                                borderRadius: '0.75rem',
                                background: 'linear-gradient(135deg, #FFC107 0%, #FFA000 100%)',
                                color: '#1A1815',
                                border: 'none',
                                fontWeight: 800,
                                fontSize: '1rem',
                                cursor: generating ? 'not-allowed' : 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '0.6rem',
                                boxShadow: '0 4px 16px rgba(255, 193, 7, 0.35)',
                                transition: 'all 0.2s',
                                opacity: generating ? 0.7 : 1
                            }}
                        >
                            {generating ? (
                                <>
                                    <RefreshCw size={18} style={{ animation: 'spin 1s linear infinite' }} />
                                    Generating Code...
                                </>
                            ) : (
                                <>
                                    <Sparkles size={18} />
                                    Generate 6-Digit Session Code
                                </>
                            )}
                        </button>
                    </div>
                </div>
            ) : (
                /* Generated Code Display Area */
                <div style={{
                    padding: '1.75rem',
                    borderRadius: '1rem',
                    background: 'rgba(255, 193, 7, 0.05)',
                    border: '1px solid rgba(255, 193, 7, 0.3)',
                    animation: 'fadeIn 0.3s ease-in-out'
                }}>
                    <div style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        flexWrap: 'wrap',
                        gap: '0.5rem',
                        marginBottom: '1rem'
                    }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <CheckCircle2 size={18} color="#FFC107" />
                            <span style={{ fontSize: '0.85rem', fontWeight: 800, color: '#FFC107', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                                Session Active &amp; Ready
                            </span>
                            {customerRef && (
                                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 600 }}>
                                    • {customerRef}
                                </span>
                            )}
                        </div>

                        {timeLeft !== null && (
                            <div style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '0.4rem',
                                fontSize: '0.82rem',
                                fontWeight: 700,
                                color: timeLeft < 180 ? '#ef4444' : 'var(--text-secondary)',
                                background: 'rgba(0, 0, 0, 0.2)',
                                padding: '0.3rem 0.75rem',
                                borderRadius: '1rem'
                            }}>
                                <Clock size={14} />
                                <span>Expires in: {formatCountdown(timeLeft)}</span>
                            </div>
                        )}
                    </div>

                    {/* 6-Digit Number Blocks */}
                    <div style={{
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'center',
                        gap: '0.6rem',
                        margin: '1.5rem 0'
                    }}>
                        {currentCode.split('').map((digit, idx) => (
                            <div
                                key={idx}
                                style={{
                                    width: '54px',
                                    height: '64px',
                                    background: 'var(--input-bg)',
                                    border: '2px solid #FFC107',
                                    borderRadius: '0.75rem',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    fontSize: '2.2rem',
                                    fontWeight: 900,
                                    fontFamily: 'monospace',
                                    color: 'var(--text-main)',
                                    boxShadow: '0 4px 12px rgba(255, 193, 7, 0.15)'
                                }}
                            >
                                {digit}
                            </div>
                        ))}
                    </div>

                    <p style={{ textAlign: 'center', fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
                        Provide this 6-digit code to the customer or share the direct link below.
                    </p>

                    {/* Action Bar */}
                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
                        gap: '0.75rem',
                        marginBottom: '1.5rem'
                    }}>
                        {/* Copy Code */}
                        <button
                            onClick={() => copyCode(currentCode)}
                            style={{
                                padding: '0.75rem 1rem',
                                borderRadius: '0.75rem',
                                background: copiedCode ? 'rgba(16, 185, 129, 0.2)' : 'rgba(255, 193, 7, 0.12)',
                                border: `1px solid ${copiedCode ? '#10B981' : 'rgba(255, 193, 7, 0.3)'}`,
                                color: copiedCode ? '#10B981' : '#FFC107',
                                fontWeight: 700,
                                fontSize: '0.9rem',
                                cursor: 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                gap: '0.5rem',
                                transition: 'all 0.2s'
                            }}
                        >
                            {copiedCode ? <><Check size={16} /> Code Copied!</> : <><Copy size={16} /> Copy Code</>}
                        </button>

                        {/* Copy Link */}
                        <button
                            onClick={() => copyLink(clientInviteUrl)}
                            style={{
                                padding: '0.75rem 1rem',
                                borderRadius: '0.75rem',
                                background: copiedLink ? 'rgba(16, 185, 129, 0.2)' : 'rgba(255, 255, 255, 0.08)',
                                border: `1px solid ${copiedLink ? '#10B981' : 'var(--glass-border)'}`,
                                color: copiedLink ? '#10B981' : 'var(--text-main)',
                                fontWeight: 700,
                                fontSize: '0.9rem',
                                cursor: 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                gap: '0.5rem',
                                transition: 'all 0.2s'
                            }}
                        >
                            {copiedLink ? <><Check size={16} /> Link Copied!</> : <><Share2 size={16} /> Copy Client Link</>}
                        </button>

                        {/* Show QR Code */}
                        <button
                            onClick={() => setShowQR(!showQR)}
                            style={{
                                padding: '0.75rem 1rem',
                                borderRadius: '0.75rem',
                                background: showQR ? 'rgba(59, 130, 246, 0.2)' : 'rgba(255, 255, 255, 0.08)',
                                border: `1px solid ${showQR ? '#3B82F6' : 'var(--glass-border)'}`,
                                color: showQR ? '#3B82F6' : 'var(--text-main)',
                                fontWeight: 700,
                                fontSize: '0.9rem',
                                cursor: 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                gap: '0.5rem',
                                transition: 'all 0.2s'
                            }}
                        >
                            <QrCode size={16} /> {showQR ? 'Hide QR' : 'Show Mobile QR'}
                        </button>

                        {/* Launch Officer Monitor */}
                        <button
                            onClick={() => navigate(`/tenant/${currentCode}`)}
                            style={{
                                padding: '0.75rem 1rem',
                                borderRadius: '0.75rem',
                                background: 'linear-gradient(135deg, #7C3AED 0%, #6D28D9 100%)',
                                border: 'none',
                                color: '#ffffff',
                                fontWeight: 700,
                                fontSize: '0.9rem',
                                cursor: 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                gap: '0.5rem',
                                boxShadow: '0 4px 12px rgba(124, 58, 237, 0.3)'
                            }}
                        >
                            <Monitor size={16} /> Launch Live Monitor
                        </button>
                    </div>

                    {/* QR Code Presentation Tray */}
                    {showQR && (
                        <div style={{
                            padding: '1.5rem',
                            borderRadius: '0.85rem',
                            background: 'var(--input-bg)',
                            border: '1px solid var(--nav-border)',
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            gap: '1rem',
                            marginBottom: '1rem'
                        }}>
                            <div style={{
                                background: '#FFFFFF',
                                padding: '1rem',
                                borderRadius: '0.75rem',
                                display: 'inline-block'
                            }}>
                                <img
                                    src={`https://api.qrserver.com/v1/create-qr-code/?size=180x180&data=${encodeURIComponent(clientInviteUrl)}`}
                                    alt="Client Join QR Code"
                                    style={{ width: '180px', height: '180px', display: 'block' }}
                                    onError={(e) => {
                                        e.target.style.display = 'none';
                                        e.target.parentNode.innerHTML = `<div style="padding:1rem;color:#000;font-size:0.85rem;font-weight:700;text-align:center;">Scan URL:<br/>${clientInviteUrl}</div>`;
                                    }}
                                />
                            </div>
                            <div style={{ textAlign: 'center' }}>
                                <div style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-main)' }}>
                                    Customer Mobile Scan
                                </div>
                                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                                    Direct the client to scan this QR code with their mobile camera to join immediately.
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Recent Generated Sessions Table */}
            {recentSessions.length > 0 && (
                <div style={{ marginTop: '2rem', borderTop: '1px solid var(--glass-border)', paddingTop: '1.5rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                        <div style={{ fontSize: '0.88rem', fontWeight: 800, color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <Clock size={16} color="var(--accent)" />
                            Recent Session Codes Generated
                        </div>
                        <button
                            onClick={() => { setRecentSessions([]); localStorage.removeItem('shadow_officer_sessions'); }}
                            style={{ background: 'none', border: 'none', color: 'var(--text-muted)', fontSize: '0.78rem', cursor: 'pointer' }}
                        >
                            Clear History
                        </button>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                        {recentSessions.map((s) => (
                            <div
                                key={s.code}
                                style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'space-between',
                                    padding: '0.75rem 1rem',
                                    borderRadius: '0.75rem',
                                    background: s.code === currentCode ? 'rgba(255, 193, 7, 0.08)' : 'var(--input-bg)',
                                    border: `1px solid ${s.code === currentCode ? '#FFC107' : 'var(--glass-border)'}`,
                                    flexWrap: 'wrap',
                                    gap: '0.75rem'
                                }}
                            >
                                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                                    <div style={{
                                        fontFamily: 'monospace',
                                        fontSize: '1.25rem',
                                        fontWeight: 900,
                                        color: '#FFC107',
                                        letterSpacing: '2px'
                                    }}>
                                        {s.code}
                                    </div>
                                    <div>
                                        <div style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-main)' }}>
                                            {s.customerRef}
                                        </div>
                                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                            Created at {s.createdAt} • {s.expiryMinutes} min validity
                                        </div>
                                    </div>
                                </div>

                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    <button
                                        onClick={() => copyCode(s.code)}
                                        title="Copy code"
                                        style={{
                                            padding: '0.4rem 0.65rem',
                                            borderRadius: '0.5rem',
                                            background: 'rgba(255, 255, 255, 0.06)',
                                            border: '1px solid var(--glass-border)',
                                            color: 'var(--text-secondary)',
                                            fontSize: '0.78rem',
                                            cursor: 'pointer',
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: '0.3rem'
                                        }}
                                    >
                                        <Copy size={13} /> Copy
                                    </button>
                                    <button
                                        onClick={() => copyLink(`${window.location.origin}/verify/${s.code}`)}
                                        title="Copy Client Link"
                                        style={{
                                            padding: '0.4rem 0.65rem',
                                            borderRadius: '0.5rem',
                                            background: 'rgba(255, 255, 255, 0.06)',
                                            border: '1px solid var(--glass-border)',
                                            color: 'var(--text-secondary)',
                                            fontSize: '0.78rem',
                                            cursor: 'pointer',
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: '0.3rem'
                                        }}
                                    >
                                        <Share2 size={13} /> Link
                                    </button>
                                    <button
                                        onClick={() => navigate(`/tenant/${s.code}`)}
                                        title="Launch Monitor"
                                        style={{
                                            padding: '0.4rem 0.75rem',
                                            borderRadius: '0.5rem',
                                            background: 'rgba(124, 58, 237, 0.15)',
                                            border: '1px solid rgba(124, 58, 237, 0.3)',
                                            color: '#A78BFA',
                                            fontWeight: 700,
                                            fontSize: '0.78rem',
                                            cursor: 'pointer',
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: '0.3rem'
                                        }}
                                    >
                                        <Monitor size={13} /> Monitor
                                    </button>
                                    <button
                                        onClick={() => removeSession(s.code)}
                                        title="Dismiss"
                                        style={{
                                            background: 'none',
                                            border: 'none',
                                            color: 'var(--text-muted)',
                                            cursor: 'pointer',
                                            padding: '0.3rem',
                                            display: 'flex',
                                            alignItems: 'center'
                                        }}
                                    >
                                        <X size={14} />
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};

export default SessionCodeGenerator;

