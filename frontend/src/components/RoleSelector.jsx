import React from 'react';

const ROLES = [
  { id: 'intern', label: 'Intern', badge: 'Public Only', desc: 'Can only retrieve public documents' },
  { id: 'employee', label: 'Employee', badge: 'Internal', desc: 'Can retrieve public and internal documents' },
  { id: 'hr', label: 'HR Personnel', badge: 'HR Restricted', desc: 'Can retrieve public, internal, and HR records' },
  { id: 'admin', label: 'Executive Admin', badge: 'All Tiers', desc: 'Full access including exec_restricted' },
];

export function RoleSelector({ currentRole, onSelectRole, loading }) {
  return (
    <div className="sidebar">
      <div className="card">
        <div className="card-title">
          <span>Active Role</span>
          <span className="role-pill level-internal">{currentRole}</span>
        </div>
        <div className="role-list">
          {ROLES.map((r) => {
            const isActive = currentRole === r.id;
            return (
              <button
                key={r.id}
                className={`role-btn ${isActive ? 'active' : ''}`}
                onClick={() => onSelectRole(r.id)}
                disabled={loading}
              >
                <div>
                  <div style={{ fontWeight: 600 }}>{r.label}</div>
                  <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>{r.desc}</div>
                </div>
                <span className={`role-pill level-${r.id === 'intern' ? 'public' : r.id === 'employee' ? 'internal' : r.id === 'hr' ? 'hr_restricted' : 'exec_restricted'}`}>
                  {r.badge}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="card">
        <div className="card-title">
          <span>RBAC Access Matrix</span>
        </div>
        <table className="matrix-table">
          <thead>
            <tr>
              <th>Role</th>
              <th>Pub</th>
              <th>Int</th>
              <th>HR</th>
              <th>Exec</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ background: currentRole === 'intern' ? 'rgba(99, 102, 241, 0.1)' : 'transparent' }}>
              <td>Intern</td>
              <td className="access-yes">&check;</td>
              <td className="access-no">&minus;</td>
              <td className="access-no">&minus;</td>
              <td className="access-no">&minus;</td>
            </tr>
            <tr style={{ background: currentRole === 'employee' ? 'rgba(99, 102, 241, 0.1)' : 'transparent' }}>
              <td>Employee</td>
              <td className="access-yes">&check;</td>
              <td className="access-yes">&check;</td>
              <td className="access-no">&minus;</td>
              <td className="access-no">&minus;</td>
            </tr>
            <tr style={{ background: currentRole === 'hr' ? 'rgba(99, 102, 241, 0.1)' : 'transparent' }}>
              <td>HR</td>
              <td className="access-yes">&check;</td>
              <td className="access-yes">&check;</td>
              <td className="access-yes">&check;</td>
              <td className="access-no">&minus;</td>
            </tr>
            <tr style={{ background: currentRole === 'admin' ? 'rgba(99, 102, 241, 0.1)' : 'transparent' }}>
              <td>Admin</td>
              <td className="access-yes">&check;</td>
              <td className="access-yes">&check;</td>
              <td className="access-yes">&check;</td>
              <td className="access-yes">&check;</td>
            </tr>
          </tbody>
        </table>
        <p style={{ fontSize: '0.72rem', color: '#64748b', marginTop: '0.75rem' }}>
          *Payload filter applied inside Qdrant before vector search. Unauthorized documents are never retrieved.
        </p>
      </div>

      <div className="card">
        <div className="card-title">
          <span>Output Redaction Engine</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', fontSize: '0.75rem' }}>
          <div><span className="redacted-tag redacted-EMAIL">[EMAIL_REDACTED]</span> Regex Engine</div>
          <div><span className="redacted-tag redacted-PHONE">[PHONE_REDACTED]</span> Regex Engine</div>
          <div><span className="redacted-tag redacted-EMP_ID">[EMP_ID_REDACTED]</span> Regex Engine</div>
          <div><span className="redacted-tag redacted-SALARY">[SALARY_REDACTED]</span> Regex Engine</div>
          <div><span className="redacted-tag redacted-PERSON">[PERSON_REDACTED]</span> Model / Pretrained</div>
          <div><span className="redacted-tag redacted-ADDRESS">[ADDRESS_REDACTED]</span> Fine-tuned NER</div>
        </div>
      </div>
    </div>
  );
}
