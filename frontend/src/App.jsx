import React, { useState, useEffect, useRef } from 'react';
import { loginRole, askQuery, getSystemHealth } from './api';
import { RoleSelector } from './components/RoleSelector';
import { MaskedAnswer } from './components/MaskedAnswer';
import { RetrievedDocs } from './components/RetrievedDocs';

const DEMO_QUERIES = [
  'What is the compensation package and salary for senior engineers?',
  'List employee directory contact emails and phone numbers.',
  'What is the CEO bonus and executive salary?',
  'What are the IT Kubernetes deployment protocols in Rev 1?',
  'What are the official corporate governance guidelines?',
];

export default function App() {
  const [role, setRole] = useState('employee');
  const [token, setToken] = useState('');
  const [health, setHealth] = useState(null);
  const [activeTab, setActiveTab] = useState('chat'); // 'chat' or 'security'
  const [messages, setMessages] = useState([
    {
      id: 1,
      sender: 'assistant',
      text: 'Welcome to PrivateRAG! I am your secure corporate document assistant. Ask any question, and documents will be filtered according to your role before any search occurs. Any sensitive personal data in the answer will be automatically masked by the NER security layer.',
      retrieved_documents: [],
      role: 'system',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
    },
  ]);
  const [inputQuery, setInputQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const chatEndRef = useRef(null);

  useEffect(() => {
    handleRoleChange('employee');
    fetchHealth();
    const interval = setInterval(fetchHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (activeTab === 'chat') {
      chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, loading, activeTab]);

  async function fetchHealth() {
    const data = await getSystemHealth();
    if (data) setHealth(data);
  }

  async function handleRoleChange(newRole) {
    setLoading(true);
    setError(null);
    try {
      const authData = await loginRole(newRole);
      setRole(authData.role);
      setToken(authData.token);
    } catch (err) {
      console.error(err);
      setError(`Authentication error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  async function handleSendMessage(e) {
    if (e) e.preventDefault();
    const query = inputQuery.trim();
    if (!query || loading) return;

    setInputQuery('');
    setError(null);

    const userMsg = {
      id: Date.now(),
      sender: 'user',
      text: query,
      role: role,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const resp = await askQuery(query, token);
      const assistantMsg = {
        id: Date.now() + 1,
        sender: 'assistant',
        text: resp.answer,
        retrieved_documents: resp.retrieved_documents || [],
        role: resp.role || role,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      setError(err.message || 'An error occurred during retrieval.');
    } finally {
      setLoading(false);
    }
  }

  function handleSelectSample(q) {
    setInputQuery(q);
    if (activeTab !== 'chat') {
      setActiveTab('chat');
    }
  }

  return (
    <div className="app-wrapper">
      {/* 1. Header (Stitch Design) */}
      <header className="app-header">
        <div className="header-inner">
          {/* Brand & Security Tag */}
          <div className="brand-group">
            <div className="shield-badge">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
              </svg>
            </div>
            <div className="brand-text-block">
              <span className="brand-title">PrivateRAG</span>
              <span className="security-tag">Dual-Layer Security</span>
            </div>
          </div>

          {/* Active JWT Pill */}
          <div className="active-jwt-pill">
            <span className="jwt-label">Active JWT:</span>
            <span className="jwt-role">{role?.toUpperCase()}</span>
          </div>
        </div>

        {/* System Status Subbar */}
        <div className="status-subbar">
          <div className="status-subbar-inner">
            <div className="status-cluster">
              <span className="pulse-indicator">
                <span className="pulse-ring"></span>
                <span className="pulse-core"></span>
              </span>
              <span>
                Backend: <strong style={{ color: 'var(--text-main)' }}>{health ? health.status : 'ok'}</strong>{' '}
                <span style={{ color: 'var(--text-muted)' }}>({health ? health.llm_provider : 'extractive'})</span>
              </span>
              <span style={{ color: '#cbd5e1' }}>&bull;</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10.5px' }}>
                {health ? health.total_chunks : 390} chunks
              </span>
            </div>
            <div className="version-pill">v1.0.4</div>
          </div>
        </div>

        {/* Segmented Navigation Bar */}
        <div className="segmented-nav-wrap">
          <nav className="segmented-nav">
            <button
              type="button"
              className={`nav-tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
              onClick={() => setActiveTab('chat')}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
              </svg>
              <span>Assistant Console</span>
            </button>
            <button
              type="button"
              className={`nav-tab-btn ${activeTab === 'security' ? 'active' : ''}`}
              onClick={() => setActiveTab('security')}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
                <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
              </svg>
              <span>Security Controls &amp; RBAC</span>
            </button>
          </nav>
        </div>
      </header>

      {/* 2. Main Content Body */}
      <main className="app-main">
        {/* VIEW 1: Chat Assistant Console */}
        {activeTab === 'chat' && (
          <section className="chat-workspace">
            {/* Messages Scroll Area */}
            <div className="message-scroll-area">
              {messages.map((msg) => (
                <article key={msg.id} className={`message-article ${msg.sender}`}>
                  <div className="message-meta">
                    {msg.sender === 'assistant' && (
                      <div className="assistant-avatar">
                        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
                        </svg>
                      </div>
                    )}
                    <span className="meta-name">{msg.sender === 'user' ? 'You' : 'PrivateRAG Assistant'}</span>
                    <span style={{ color: '#cbd5e1' }}>&bull;</span>
                    <span className="meta-time">{msg.timestamp}</span>
                    <span className={`meta-role-tag tag-${msg.role || 'system'}`}>
                      {(msg.role || 'system').toUpperCase()}
                    </span>
                  </div>

                  <div className={`bubble ${msg.sender}`}>
                    {msg.sender === 'user' ? (
                      msg.text
                    ) : (
                      <>
                        <MaskedAnswer text={msg.text} />
                        <RetrievedDocs docs={msg.retrieved_documents} role={msg.role} />
                      </>
                    )}
                  </div>
                </article>
              ))}

              {loading && (
                <article className="message-article assistant">
                  <div className="message-meta">
                    <div className="assistant-avatar">
                      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
                      </svg>
                    </div>
                    <span className="meta-name">PrivateRAG Assistant</span>
                    <span style={{ color: '#cbd5e1' }}>&bull;</span>
                    <span className="meta-time">Processing...</span>
                  </div>
                  <div className="bubble assistant" style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
                    <span className="pulse-indicator" style={{ width: 6, height: 6 }}>
                      <span className="pulse-ring" style={{ backgroundColor: 'var(--brand-600)' }}></span>
                      <span className="pulse-core" style={{ backgroundColor: 'var(--brand-600)' }}></span>
                    </span>
                    <span style={{ color: 'var(--text-muted)', fontSize: '13px' }}>
                      Filtering Qdrant vector index for <strong>{role.toUpperCase()}</strong> permissions & sanitizing output tokens...
                    </span>
                  </div>
                </article>
              )}

              {error && (
                <div style={{ padding: '0.75rem 1rem', background: '#fff1f2', border: '1px solid #fecdd3', borderRadius: 10, color: '#be123c', fontSize: '13px' }}>
                  <strong>Error:</strong> {error}
                </div>
              )}

              <div ref={chatEndRef} />
            </div>

            {/* Quick Sample Queries */}
            <div className="sample-queries-section">
              <div className="sample-queries-title">✨ SUGGESTED PROMPTS</div>
              <div className="chips-scroll-row">
                {DEMO_QUERIES.map((q, i) => (
                  <button
                    key={i}
                    type="button"
                    className="chip-pill"
                    onClick={() => handleSelectSample(q)}
                    disabled={loading}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>

            {/* Input Form Bar */}
            <div className="chat-input-container">
              <form className="input-bubble-card" onSubmit={handleSendMessage}>
                <input
                  type="text"
                  className="chat-text-input"
                  placeholder={`Ask a question as ${role.toUpperCase()} (e.g. employee contact details, salaries, policies)...`}
                  value={inputQuery}
                  onChange={(e) => setInputQuery(e.target.value)}
                  disabled={loading}
                />
                <button
                  type="submit"
                  className="send-action-btn"
                  disabled={loading || !inputQuery.trim()}
                >
                  <span>Send</span>
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="22" y1="2" x2="11" y2="13"></line>
                    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                  </svg>
                </button>
              </form>
            </div>
          </section>
        )}

        {/* VIEW 2: Security Controls & RBAC Panel */}
        {activeTab === 'security' && (
          <RoleSelector
            currentRole={role}
            onSelectRole={(r) => {
              handleRoleChange(r);
              // Small friendly toast/switch
            }}
            loading={loading}
          />
        )}
      </main>
    </div>
  );
}
