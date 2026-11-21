import React, { useState } from 'react';
import { login } from '../api';

// Prototype demo accounts (passwords documented in README).
// Roles are assigned by the SERVER from the user store — the client
// only supplies credentials, never a privilege level.
const DEMO_ACCOUNTS = [
  { username: 'intern', password: 'Intern#2026', tier: 'Public only' },
  { username: 'employee', password: 'Employee#2026', tier: 'Public + Internal' },
  { username: 'hr', password: 'HrSpec#2026', tier: '+ HR restricted' },
  { username: 'admin', password: 'Admin#2026', tier: 'All tiers' },
];

export function LoginPanel({ onLogin, notice }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!username.trim() || !password || loading) return;
    setLoading(true);
    setError(null);
    try {
      const authData = await login(username.trim(), password);
      onLogin(authData);
    } catch (err) {
      setError(err.message || 'Login failed.');
    } finally {
      setLoading(false);
    }
  }

  function fillDemo(account) {
    setUsername(account.username);
    setPassword(account.password);
    setError(null);
  }

  return (
    <div className="login-screen">
      <div className="login-card">
        <div className="login-brand">
          <div className="shield-badge">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
            </svg>
          </div>
          <div className="brand-text-block">
            <span className="brand-title">PrivateRAG</span>
            <span className="security-tag">Dual-Layer Security</span>
          </div>
        </div>

        <h2 className="login-heading">Sign in to continue</h2>
        {notice && <div className="login-notice">{notice}</div>}
        <p className="login-subtext">
          Your clearance level is assigned by the server after credential
          verification — it cannot be selected or requested by the client.
        </p>

        <form onSubmit={handleSubmit} className="login-form">
          <label className="login-label" htmlFor="login-username">Username</label>
          <input
            id="login-username"
            type="text"
            className="login-input"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="e.g. employee"
            autoComplete="username"
            disabled={loading}
          />
          <label className="login-label" htmlFor="login-password">Password</label>
          <input
            id="login-password"
            type="password"
            className="login-input"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="••••••••••"
            autoComplete="current-password"
            disabled={loading}
          />
          {error && <div className="login-error">{error}</div>}
          <button
            type="submit"
            className="send-action-btn login-submit"
            disabled={loading || !username.trim() || !password}
          >
            {loading ? 'Authenticating…' : 'Sign in'}
          </button>
        </form>

        <div className="demo-accounts">
          <div className="sample-queries-title">DEMO ACCOUNTS (COURSE PROTOTYPE)</div>
          {DEMO_ACCOUNTS.map((a) => (
            <button
              key={a.username}
              type="button"
              className="demo-account-row"
              onClick={() => fillDemo(a)}
              disabled={loading}
            >
              <span className="demo-account-name">{a.username}</span>
              <span className="demo-account-tier">{a.tier}</span>
            </button>
          ))}
          <p className="demo-account-note">Click an account to fill its credentials.</p>
        </div>
      </div>
    </div>
  );
}