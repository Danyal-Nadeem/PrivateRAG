import React from 'react';

const ROLES = [
  {
    id: 'intern',
    label: 'Intern',
    badge: 'PUBLIC ONLY',
    badgeClass: 'tag-intern',
    desc: 'Can only retrieve public documents',
  },
  {
    id: 'employee',
    label: 'Employee',
    badge: 'INTERNAL',
    badgeClass: 'tag-employee',
    desc: 'Can retrieve public and internal documents',
  },
  {
    id: 'hr',
    label: 'HR Personnel',
    badge: 'HR RESTRICTED',
    badgeClass: 'tag-hr',
    desc: 'Can retrieve public, internal, and HR records',
  },
  {
    id: 'admin',
    label: 'Executive Admin',
    badge: 'ALL TIERS',
    badgeClass: 'tag-admin',
    desc: 'Full access including exec_restricted',
  },
];

export function RoleSelector({ currentRole, onSelectRole, loading }) {
  return (
    <div className="security-controls-panel">
      {/* 1. Active Role Selection */}
      <div>
        <div className="panel-section-title">
          <span className="section-h2">Active Role</span>
          <span className={`meta-role-tag tag-${currentRole}`}>{currentRole.toUpperCase()}</span>
        </div>
        <div className="role-cards-grid">
          {ROLES.map((r) => {
            const isSelected = currentRole === r.id;
            return (
              <div
                key={r.id}
                className={`role-card-item ${isSelected ? 'selected' : ''}`}
                onClick={() => !loading && onSelectRole(r.id)}
                role="button"
                tabIndex={0}
              >
                <div className="role-card-left">
                  <div className="role-name-row">
                    <span className="role-title">{r.label}</span>
                    {isSelected && <span className="role-active-dot"></span>}
                  </div>
                  <span className="role-desc">{r.desc}</span>
                </div>
                <span className={`role-tier-badge ${r.badgeClass}`}>{r.badge}</span>
              </div>
            );
          })}
        </div>
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
            <span className="detector-label">Model / Pretrained</span>
          </div>
          <div className="redaction-item-row">
            <span className="redacted-tag redacted-ADDRESS">[ADDRESS_REDACTED]</span>
            <span className="detector-label">Fine-tuned NER</span>
          </div>
        </div>
      </div>
    </div>
  );
}
