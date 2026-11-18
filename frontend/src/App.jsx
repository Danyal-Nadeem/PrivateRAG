import React, { useState, useEffect, useRef } from 'react';
import { loginRole, askQuery, getSystemHealth } from './api';
import { RoleSelector } from './components/RoleSelector';
import { MaskedAnswer } from './components/MaskedAnswer';
import { RetrievedDocs } from './components/RetrievedDocs';

const DEMO_QUERIES = [
  'What is the compensation package and salary for senior engineers?',
  'List employee directory contact emails and phone numbers.',
  'What are the official company public holidays?',
  'What are the executive acquisition and board strategies?',
];

export default function App() {
  const [role, setRole] = useState('employee');
  const [token, setToken] = useState('');
  const [health, setHealth] = useState(null);
  const [messages, setMessages] = useState([
    {
      sender: 'assistant',
      text: 'Welcome to PrivateRAG! I am your secure corporate document assistant. Ask any question, and documents will be filtered according to your role before any search occurs. Any sensitive personal data in the answer will be automatically masked by the NER security layer.',
      retrieved_documents: [],
      role: 'system',
      timestamp: new Date().toLocaleTimeString(),
    },
  ]);
  const [inputQuery, setInputQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const chatEndRef = useRef(null);

  // Initialize role and check health
  useEffect(() => {
    handleRoleChange('employee');
    fetchHealth();
    const interval = setInterval(fetchHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

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
      setError(`Failed to authenticate as ${newRole}: ${err.message}`);
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
      sender: 'user',
      text: query,
      role: role,
      timestamp: new Date().toLocaleTimeString(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const resp = await askQuery(query, token);
      const assistantMsg = {
        sender: 'assistant',
        text: resp.answer,
        retrieved_documents: resp.retrieved_documents || [],
        role: resp.role || role,
        timestamp: new Date().toLocaleTimeString(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      setError(err.message || 'An error occurred during retrieval.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-container">
      {/* Top Navbar */}
      <header className="navbar">
        <div className="nav-brand">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#6366f1" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
          </svg>
          <span>PrivateRAG</span>
          <span className="brand-badge">Dual-Layer Security</span>
        </div>

        <div className="system-status">
          <div className="status-indicator">
            <span className="pulse-dot"></span>
            <span>
              {health ? (
                <>Backend: <strong>{health.status}</strong> ({health.llm_provider}) &bull; {health.total_chunks} chunks</>
              ) : (
                'Connecting to API...'
              )}
            </span>
          </div>
          <div>
            Active JWT: <span className="role-pill level-internal">{role}</span>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="main-content">
        {/* Role & Access Controls Sidebar */}
        <RoleSelector
          currentRole={role}
          onSelectRole={handleRoleChange}
          loading={loading}
        />

        {/* Chat Section */}
        <section className="chat-section">
          <div className="chat-history">
            {messages.map((msg, idx) => (
              <div key={idx} className={`message-item ${msg.sender}`}>
                <div className="message-header">
                  <span>{msg.sender === 'user' ? 'You' : 'PrivateRAG Assistant'}</span>
                  <span>&bull;</span>
                  <span>{msg.timestamp}</span>
                  {msg.role && (
                    <span className={`role-pill level-${msg.role === 'intern' ? 'public' : msg.role === 'employee' ? 'internal' : msg.role === 'hr' ? 'hr_restricted' : 'exec_restricted'}`}>
                      {msg.role}
                    </span>
                  )}
                </div>

                <div className="message-bubble">
                  {msg.sender === 'user' ? (
                    msg.text
                  ) : (
                    <>
                      <MaskedAnswer text={msg.text} />
                      <RetrievedDocs docs={msg.retrieved_documents} role={msg.role} />
                    </>
                  )}
                </div>
              </div>
            ))}

            {loading && (
              <div className="message-item assistant">
                <div className="message-header">
                  <span>PrivateRAG Assistant</span>
                  <span>&bull;</span>
                  <span>Processing...</span>
                </div>
                <div className="message-bubble" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <div className="pulse-dot" style={{ backgroundColor: 'var(--primary)', boxShadow: '0 0 8px var(--primary)' }}></div>
                  <span style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                    Filtering Qdrant vector index ({role} permissions) & sanitizing output tokens...
                  </span>
                </div>
              </div>
            )}

            {error && (
              <div style={{ padding: '0.75rem 1rem', background: 'rgba(239, 68, 68, 0.15)', border: '1px solid rgba(239, 68, 68, 0.4)', borderRadius: 8, color: '#f87171', fontSize: '0.875rem' }}>
                &times; {error}
              </div>
            )}

            <div ref={chatEndRef} />
          </div>

          {/* Prompt Chips */}
          <div className="prompt-chips">
            <span style={{ fontSize: '0.75rem', color: '#64748b', alignSelf: 'center', marginRight: '0.25rem' }}>Sample:</span>
            {DEMO_QUERIES.map((q, i) => (
              <button
                key={i}
                type="button"
                className="chip-btn"
                onClick={() => setInputQuery(q)}
                disabled={loading}
              >
                {q}
              </button>
            ))}
          </div>

          {/* Input Form */}
          <form className="chat-input-bar" onSubmit={handleSendMessage}>
            <input
              type="text"
              className="chat-input"
              placeholder={`Ask a question as ${role.toUpperCase()} (e.g. employee contact details, salaries, policies)...`}
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              disabled={loading}
            />
            <button type="submit" className="send-btn" disabled={loading || !inputQuery.trim()}>
              <span>Send</span>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="22" y1="2" x2="11" y2="13"></line>
                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
              </svg>
            </button>
          </form>
        </section>
      </main>
    </div>
  );
}
