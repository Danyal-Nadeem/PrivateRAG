import React from 'react';

const REDACTION_PATTERN = /(\[(?:PERSON|EMAIL|PHONE|EMP_ID|ADDRESS|SALARY|[A-Z_]+)_REDACTED\])/g;

function renderFormattedText(text, keyPrefix) {
  // Support **bold** markdown rendering
  const segments = text.split(/(\*\*[^*]+\*\*)/g);
  return segments.map((seg, idx) => {
    if (seg.startsWith('**') && seg.endsWith('**') && seg.length > 4) {
      return (
        <strong key={`${keyPrefix}-b-${idx}`} style={{ color: 'var(--text-main)', fontWeight: 600 }}>
          {seg.slice(2, -2)}
        </strong>
      );
    }
    return seg;
  });
}

export function MaskedAnswer({ text }) {
  if (!text) return null;

  const parts = text.split(REDACTION_PATTERN);

  return (
    <div className="answer-text">
      {parts.map((part, index) => {
        const match = part.match(/^\[([A-Z_]+)_REDACTED\]$/);
        if (match) {
          const label = match[1];
          return (
            <span
              key={index}
              className={`redacted-tag redacted-${label}`}
              title={`Sensitive ${label} masked by PrivateRAG NLP output filter`}
            >
              🔒 {part}
            </span>
          );
        }
        return <span key={index}>{renderFormattedText(part, index)}</span>;
      })}
    </div>
  );
}
