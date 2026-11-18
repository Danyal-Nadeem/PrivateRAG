import React from 'react';

export function RetrievedDocs({ docs, role }) {
  if (!docs || docs.length === 0) return null;

  return (
    <div className="docs-inspector-card">
      <div className="docs-inspector-header">
        <span>
          Retrieved Documents ({docs.length}) &bull; Filtered for: <strong style={{ color: 'var(--brand-700)' }}>{role?.toUpperCase()}</strong>
        </span>
        <span style={{ fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Vector Sim Score</span>
      </div>
      <div className="docs-list">
        {docs.map((doc, idx) => (
          <div key={`${doc.doc_id}-${idx}`} className="doc-row">
            <div className="doc-main-info">
              <span className={`doc-tier-badge tier-${doc.access_level}`}>
                {doc.access_level?.toUpperCase()}
              </span>
              <span className="doc-id-label">{doc.doc_id}</span>
              <span className="doc-title-label" title={doc.title}>
                {doc.title}
              </span>
            </div>
            <span className="doc-score-pill">
              {(doc.score * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
