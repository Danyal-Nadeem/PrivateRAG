import React from 'react';

export function RetrievedDocs({ docs, role }) {
  if (!docs || docs.length === 0) return null;

  return (
    <div className="docs-inspector">
      <div className="docs-header">
        <span>Retrieved Documents ({docs.length}) &bull; Filtered for: <strong>{role}</strong></span>
        <span>Vector Sim Score</span>
      </div>
      <div className="doc-chip-list">
        {docs.map((doc, idx) => (
          <div key={`${doc.doc_id}-${idx}`} className="doc-chip">
            <div className="doc-chip-info">
              <span className={`role-pill level-${doc.access_level}`}>
                {doc.access_level}
              </span>
              <span className="doc-id">{doc.doc_id}</span>
              <span className="doc-title" title={doc.title}>
                {doc.title}
              </span>
            </div>
            <span className="doc-score">
              {(doc.score * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
