import React from 'react';

const REDACTION_PATTERN = /(\[(?:PERSON|EMAIL|PHONE|EMP_ID|ADDRESS|SALARY|[A-Z_]+)_REDACTED\])/g;

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
        return <span key={index}>{part}</span>;
      })}
    </div>
  );
}
