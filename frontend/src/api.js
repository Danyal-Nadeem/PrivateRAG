/**
 * API client for PrivateRAG backend.
 * Base URL defaults to window.location.origin (which Vite proxies to http://localhost:8000)
 * or explicit http://localhost:8000 if running standalone.
 */

const API_BASE = import.meta.env.VITE_API_URL || '';

export async function login(username, password) {
  const resp = await fetch(`${API_BASE}/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });

  if (!resp.ok) {
    if (resp.status === 401) {
      throw new Error('Invalid username or password.');
    }
    const errorText = await resp.text();
    throw new Error(`Login failed (${resp.status}): ${errorText}`);
  }

  const data = await resp.json();
  localStorage.setItem('privaterag_token', data.token);
  localStorage.setItem('privaterag_role', data.role);
  localStorage.setItem('privaterag_username', data.username);
  return data;
}

export function logout() {
  localStorage.removeItem('privaterag_token');
  localStorage.removeItem('privaterag_role');
  localStorage.removeItem('privaterag_username');
}

export async function askQuery(query, token) {
  const authToken = token || localStorage.getItem('privaterag_token');
  if (!authToken) {
    throw new Error('No authentication token found. Please log in first.');
  }

  const resp = await fetch(`${API_BASE}/ask`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${authToken}`,
    },
    body: JSON.stringify({ query }),
  });

  if (!resp.ok) {
    const errorText = await resp.text();
    throw new Error(`Query failed (${resp.status}): ${errorText}`);
  }

  return await resp.json();
}

export async function getSystemHealth() {
  try {
    const resp = await fetch(`${API_BASE}/health`);
    if (!resp.ok) return null;
    return await resp.json();
  } catch (err) {
    console.warn('Backend /health unreachable:', err);
    return null;
  }
}
