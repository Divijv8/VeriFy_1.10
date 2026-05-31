import re
import string
from collections import Counter
import numpy as np

def remove_special_chars(x):
    pattern = r'[^\w\s]'
    return re.sub(pattern, '', str(x))

def tokenize_training_style(text):
    # EXACT same as training: remove_special_chars then split (no lowercasing)
    cleaned = remove_special_chars(text)
    return cleaned.split()

def top_n_tokens_from_sequence(tokens, n=10):
    cnt = Counter(tokens)
    return cnt.most_common(n)

def type_token_ratio(tokens):
    if not tokens: return 0.0
    return len(set(tokens)) / len(tokens)

def avg_word_length(tokens):
    if not tokens: return 0.0
    return np.mean([len(w) for w in tokens])

def avg_sentence_length(text):
    # naive sentence split by punctuation
    sents = re.split(r'[.!?]+', text)
    sents = [s.strip() for s in sents if s.strip()]
    if not sents: return 0.0
    return np.mean([len(s.split()) for s in sents])

def punctuation_density(text):
    if not text: return 0.0
    punct_count = sum(1 for ch in text if ch in string.punctuation)
    return punct_count / max(1, len(text.split()))

def sensational_word_count(tokens, sensational_set):
    return sum(1 for t in tokens if t.lower() in sensational_set)

# simple date extraction helper
_DATE_PATTERNS = [
    r'\b(?:\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{2,4})\b',
    r'\b(?:\d{1,2}/\d{1,2}/\d{2,4})\b',
    r'\b(?:\d{4}-\d{1,2}-\d{1,2})\b',
    r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
    r'\b(?:today|yesterday|tomorrow|last week|next week|last month|next month)\b'
]
def extract_dates(text):
    t = text.lower()
    found = []
    for p in _DATE_PATTERNS:
        found += re.findall(p, t, flags=re.IGNORECASE)
    return list(dict.fromkeys(found))  # dedupe preserving order
