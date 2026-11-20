import React from 'react';

const ROLE_INFO = {
  intern: { label: 'Intern', badge: 'PUBLIC ONLY', badgeClass: 'tag-intern', desc: 'Can only retrieve public documents' },
  employee: { label: 'Employee', badge: 'INTERNAL', badgeClass: 'tag-employee', desc: 'Can retrieve public and internal documents' },
  hr: { label: 'HR Personnel', badge: 'HR RESTRICTED', badgeClass: 'tag-hr', desc: 'Can retrieve public, internal, and HR records' },
  admin: { label: 'Executive Admin', badge: 'ALL TIERS', badgeClass: 'tag-admin', desc: 'Full access including exec_restricted' },
};

export function SecurityPanel({ currentRole, username, onLogout, nerModelLoaded }) {
  const info = ROLE_INFO[currentRole] || ROLE_INFO.intern;
  return (
    <div className="security-controls-panel">
      {/* 1. Authenticated Account (read-only — role is server-assigned) */}
      <div>
        <div className="panel-section-title">
          <span className="section-h2">Authenticated Account</span>
          <span className={`meta-role-tag tag-${currentRole}`}>{currentRole.toUpperCase()}</span>
        </div>
        <div className="role-cards-grid">
          <div className="role-card-item selected">
            <div className="role-card-left">
              <div className="role-name-row">
                <span className="role-title">{username || currentRole}</span>
                <span className="role-active-dot"></span>
              </div>
              <span className="role-desc">{info.desc}. Clearance is assigned by the server and cannot be changed from the client.</span>
            </div>
            <span className={`role-tier-badge ${info.badgeClass}`}>{info.badge}</span>
          </div>
        </div>
        <button type="button" className="logout-btn" onClick={onLogout}>
          Sign out
        </button>
      </div>

      {/* 2. RBAC Access Matrix */}
      <div className="rbac-card">
        <span className="section-h2">RBAC Access Matrix</span>
        <table className="rbac-table">
          <thead>
            <tr>
              <th style={{ textAlign: 'left' }}>Role</th>
              <th className="cell-center">Pub</th>
              <th className="cell-center">Int</th>
              <th className="cell-center">HR</th>
              <th className="cell-center">Exec</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style={{ fontFamily: 'var(--font-sans)', fontWeight: 500 }}>Intern</td>
              <td className="cell-center check-green">✓</td>
              <td className="cell-center dash-gray">-</td>
              <td className="cell-center dash-gray">-</td>
              <td className="cell-center dash-gray">-</td>
            </tr>
            <tr>
              <td style={{ fontFamily: 'var(--font-sans)', fontWeight: 500 }}>Employee</td>
              <td className="cell-center check-green">✓</td>
              <td className="cell-center check-green">✓</td>
              <td className="cell-center dash-gray">-</td>
              <td className="cell-center dash-gray">-</td>
            </tr>
            <tr>
              <td style={{ fontFamily: 'var(--font-sans)', fontWeight: 500 }}>HR</td>
              <td className="cell-center check-green">✓</td>
              <td className="cell-center check-green">✓</td>
              <td className="cell-center check-green">✓</td>
              <td className="cell-center dash-gray">-</td>
            </tr>
            <tr>
              <td style={{ fontFamily: 'var(--font-sans)', fontWeight: 500 }}>Admin</td>
              <td className="cell-center check-green">✓</td>
              <td className="cell-center check-green">✓</td>
              <td className="cell-center check-green">✓</td>
              <td className="cell-center check-green">✓</td>
            </tr>
          </tbody>
        </table>
        <p className="rbac-footnote">
          *Payload filter applied inside Qdrant before vector search. Unauthorized documents are never retrieved into memory.
        </p>
      </div>

      {/* 3. Output Redaction Engine */}
      <div className="redaction-engine-card">
        <span className="section-h2">Output Redaction Engine</span>
        {nerModelLoaded === false && (
          <p className="dlp-degraded-banner">
            ⚠ DLP model checkpoint not loaded — PERSON and ADDRESS are currently
            NOT masked (regex-only fallback). See /health and server logs.
          </p>
        )}
        <div className="redaction-grid">
          <div className="redaction-item-row">
            <span className="redacted-tag redacted-EMAIL">[EMAIL_REDACTED]</span>
            <span className="detector-label">Regex Engine</span>
          </div>
          <div className="redaction-item-row">
            <span className="redacted-tag redacted-PHONE">[PHONE_REDACTED]</span>
            <span className="detector-label">Regex Engine</span>
          </div>
          <div className="redaction-item-row">
            <span className="redacted-tag redacted-EMP_ID">[EMP_ID_REDACTED]</span>
            <span className="detector-label">Regex Engine</span>
          </div>
          <div className="redaction-item-row">
            <span className="redacted-tag redacted-SALARY">[SALARY_REDACTED]</span>
            <span className="detector-label">Regex Engine</span>
          </div>
          <div className="redaction-item-row">
            <span className="redacted-tag redacted-PERSON">[PERSON_REDACTED]</span>
            <span className="detector-label">{nerModelLoaded ? 'Fine-tuned NER' : 'Fine-tuned NER (not loaded)'}</span>
          </div>
          <div className="redaction-item-row">
            <span className="redacted-tag redacted-ADDRESS">[ADDRESS_REDACTED]</span>
            <span className="detector-label">{nerModelLoaded ? 'Fine-tuned NER' : 'Fine-tuned NER (not loaded)'}</span>
          </div>
        </div>
      </div>
    </div>
  );
}