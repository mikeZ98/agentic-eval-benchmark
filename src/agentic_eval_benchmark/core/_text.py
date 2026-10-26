"""Deterministic text primitives for the offline judge: normalization, tokens, claims.

Everything here is pure, linear-time over its input and free of models or downloads.
"""

import itertools
import re
import unicodedata
from typing import Final

_TOKEN_RE: Final = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+")
_SPACE_RE: Final = re.compile(r"\s+")
_SENTENCE_SPLIT_RE: Final = re.compile(r"(?<=[.!?;])\s+")
_BULLET_RE: Final = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
# Typographic quotes are folded to ASCII before matching (see normalize).
_QUOTE_FOLD: Final = str.maketrans({0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"'})
_EDGE_PUNCT: Final = " \t\"'`.,;:!?()[]{}" + "".join(map(chr, (0x2018, 0x2019, 0x201C, 0x201D)))

_NEGATION_CUES: Final = frozenset(
    {"not", "no", "never", "none", "neither", "nor", "cannot", "without"}
)
# Tokenization splits "isn't" into ("isn", "t") and "can't" into ("can", "t").
_CONTRACTED_NEGATIONS: Final = frozenset(
    {"isn", "wasn", "aren", "weren", "doesn", "didn", "don", "won", "can", "couldn",
     "shouldn", "wouldn", "hasn", "haven", "hadn", "mustn"}
)  # fmt: skip
# "can" is a stopword and "won" is also a content word ("who won?"): keep both as tokens.
_NON_CONTENT: Final = _NEGATION_CUES | (_CONTRACTED_NEGATIONS - {"can", "won"})

STOPWORDS: Final = frozenset(
    {"a", "an", "the", "and", "or", "but", "if", "then", "so", "of", "to", "in", "on", "at",
     "by", "for", "with", "from", "as", "into", "about", "over", "after", "before", "between",
     "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "done",
     "has", "have", "had", "having", "will", "would", "shall", "should", "may", "might",
     "must", "could", "can", "it", "its", "this", "that", "these", "those", "there", "here",
     "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "i", "you", "he",
     "she", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "our",
     "their", "also", "just", "than", "too", "very", "such", "any", "all", "each", "some",
     "both", "other", "more", "most", "s", "t", "please", "tell", "explain", "describe"}
)  # fmt: skip
"""Function words ignored when comparing content. Negations are handled separately."""

_SUFFIXES: Final = (
    "izations", "ization", "ational", "fulness", "iveness", "ements", "ement", "ments",
    "ment", "ings", "ing", "edly", "ied", "ies", "ed", "es", "ly", "s",
)  # fmt: skip


def normalize(text: str) -> str:
    """NFKC-fold, casefold, fold typographic quotes and collapse whitespace."""
    folded = unicodedata.normalize("NFKC", text).casefold().translate(_QUOTE_FOLD)
    return _SPACE_RE.sub(" ", folded).strip()


def strip_edges(text: str) -> str:
    """Drop surrounding quotes, brackets and sentence punctuation."""
    return text.strip(_EDGE_PUNCT)


def tokenize(text: str) -> list[str]:
    """Word and number tokens of the normalized text (thousands separators removed)."""
    return [t.replace(",", "") if t[0].isdigit() else t for t in _TOKEN_RE.findall(normalize(text))]


def stem(token: str) -> str:
    """Light, dependency-free suffix stripping so inflections compare equal."""
    if token[0].isdigit():
        return token
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:  # noqa: PLR2004
            base = token[: -len(suffix)]
            return base + "y" if suffix in {"ies", "ied"} else base
    return token


def content_terms(text: str) -> frozenset[str]:
    """Stemmed content tokens: stopwords and negation cues removed."""
    return frozenset(
        stem(t) for t in tokenize(text) if t not in STOPWORDS and t not in _NON_CONTENT
    )


def numbers(text: str) -> frozenset[str]:
    """Numeric tokens; a claim must not introduce numbers its evidence lacks."""
    return frozenset(t for t in tokenize(text) if t[0].isdigit())


def is_negated(text: str) -> bool:
    """Whether the text carries a negation (``not``, ``never``, ``isn't`` …)."""
    tokens = tokenize(text)
    if _NEGATION_CUES.intersection(tokens):
        return True
    return any(a in _CONTRACTED_NEGATIONS and b == "t" for a, b in itertools.pairwise(tokens))


def split_claims(text: str) -> list[str]:
    """Split text into atomic statements (sentences, clauses after ``;``, bullet lines)
    and keep those with at least one content term."""
    claims: list[str] = []
    for line in text.splitlines():
        # Strip list markers first: "1. Step" must not split into "1" and "Step".
        for part in _SENTENCE_SPLIT_RE.split(_BULLET_RE.sub("", line)):
            claim = strip_edges(part.strip())
            if claim and content_terms(claim):
                claims.append(claim)
    return claims


def coverage(needle: frozenset[str], haystack: frozenset[str]) -> float:
    """Fraction of ``needle`` terms present in ``haystack`` (``1.0`` for an empty needle)."""
    if not needle:
        return 1.0
    return len(needle & haystack) / len(needle)


def overlap_coefficient(a: frozenset[str], b: frozenset[str]) -> float:
    """``|a & b| / min(|a|, |b|)``: similarity that tolerates different chunk sizes."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def excerpt(text: str, limit: int = 80) -> str:
    """Single-line excerpt for judge reasoning."""
    flat = _SPACE_RE.sub(" ", text).strip()
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"
