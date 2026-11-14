import re
from typing import List, Dict, Tuple, Any

TOKEN_REGEX = re.compile(r"\w+|[^\w\s]")

def tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    r"""Tokenizes text using \w+|[^\w\s] and returns list of (token_str, start_char, end_char)."""
    tokens = []
    for match in TOKEN_REGEX.finditer(text):
        tokens.append((match.group(0), match.start(), match.end()))
    return tokens

def align_tokens_and_bio(
    text: str,
    entities: List[Dict[str, Any]],
    tokens_with_offsets: List[Tuple[str, int, int]]
) -> List[str]:
    """
    Given text, entity spans [{'start': int, 'end': int, 'label': str, 'text': str}],
    and tokens with offsets, assigns standard BIO tags.
    Raises ValueError if any entity boundary cuts across a token boundary.
    """
    sorted_entities = sorted(entities, key=lambda x: x["start"])
    
    for i in range(len(sorted_entities) - 1):
        if sorted_entities[i]["end"] > sorted_entities[i + 1]["start"]:
            raise ValueError(
                f"Overlapping entities detected: {sorted_entities[i]} and {sorted_entities[i+1]}"
            )
            
    token_starts = {t[1]: idx for idx, t in enumerate(tokens_with_offsets)}
    token_ends = {t[2]: idx for idx, t in enumerate(tokens_with_offsets)}
    
    ner_tags = ["O"] * len(tokens_with_offsets)
    
    for ent in sorted_entities:
        e_start = ent["start"]
        e_end = ent["end"]
        e_label = ent["label"]
        
        if e_start not in token_starts:
            raise ValueError(
                f"Entity start {e_start} for '{ent.get('text', '')}' does not align with any token start in: {text}"
            )
        if e_end not in token_ends:
            raise ValueError(
                f"Entity end {e_end} for '{ent.get('text', '')}' does not align with any token end in: {text}"
            )
            
        start_tok_idx = token_starts[e_start]
        end_tok_idx = token_ends[e_end]
        
        ner_tags[start_tok_idx] = f"B-{e_label}"
        for tok_idx in range(start_tok_idx + 1, end_tok_idx + 1):
            ner_tags[tok_idx] = f"I-{e_label}"
            
    return ner_tags
