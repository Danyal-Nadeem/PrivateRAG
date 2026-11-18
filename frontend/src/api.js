/**
 * API client for PrivateRAG backend.
 * Base URL defaults to window.location.origin (which Vite proxies to http://localhost:8000)
 * or explicit http://localhost:8000 if running standalone.
 */

const API_BASE = import.meta.env.VITE_API_URL || '';

export async function loginRole(role) {
  const resp = await fetch(`${API_BASE}/login-role`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ role }),
  });

  if (!resp.ok) {
    const errorText = await resp.text();
    throw new Error(`Login failed (${resp.status}): ${errorText}`);
  }

  const data = await resp.json();
  localStorage.setItem('privaterag_token', data.token);
  localStorage.setItem('privaterag_role', data.role);
  return data;
}

export async function askQuery(query, token) {
  const authToken = token || localStorage.getItem('privaterag_token');
  if (!authToken) {
    throw new Error('No authentication token found. Please select a role first.');
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
