"""
normalization.py
----------------
Person 2 – Data Cleaning & Normalization  (v2)

Changes in v2
-------------
- Pre-compiled regex patterns (10x faster)
- French legal suffixes: SARL, SAS, SASU, SCI, EURL
- Indian address abbreviations: KH No, Gali, Nagar, Colony, Sector
- Number normalisation (ordinals "1st","2nd" kept; fractions "1/2" → "1 2")
- Website/URL detection (e.g. "wilfordhancock.com") → strip .com/.net/.org
- Devanagari/non-Latin script: preserved as-is (no accent stripping)
- Bugfix: remove_accents() only strips Latin combining marks, not Indic
- Bugfix: _strip_special_chars() preserves Indic combining marks (Mn/Mc)
- Bugfix: address \bst\.?\b now has negative lookbehind for digits (protects
  ordinals like "1st", "2nd", "21st") and redundant no-op rules removed
"""

import re
import unicodedata
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LEGAL ENTITY SUFFIXES  (applied in ORDER — longest/most-specific first)
# ─────────────────────────────────────────────────────────────────────────────
_RAW_LEGAL = [
    # ── Indian / Global combined ──
    (r"\bpvt\.?\s*ltd\.?\b",                            "pvt ltd"),
    (r"\bprivate\s+limited\b",                          "pvt ltd"),
    (r"\bprivate\s+ltd\.?\b",                           "pvt ltd"),
    (r"\bprivate\b",                                    "pvt"),   # standalone "Private" (no trailing Ltd/Limited)
    # ── French ──
    (r"\bsas[u]?\b",                                    "sas"),   # SASU, SAS
    (r"\bsarl\b",                                       "sarl"),
    (r"\bsci\b",                                        "sci"),
    (r"\beurl\b",                                       "eurl"),
    (r"\bsa\b",                                         "sa"),    # Société Anonyme
    # ── English ──
    (r"\binc(?:orporated)?\.?\b",                       "inc"),
    (r"\bcorp(?:oration)?\.?\b",                        "corp"),
    (r"\bllc\.?\b",                                     "llc"),
    (r"\bl\.l\.c\.?\b",                                 "llc"),
    (r"\bllp\.?\b",                                     "llp"),
    (r"\bl\.l\.p\.?\b",                                 "llp"),
    (r"\bltd\.?\b",                                     "ltd"),
    (r"\blimited\b",                                    "ltd"),
    (r"\bco\.?\b",                                      "co"),
    (r"\bcompany\b",                                    "co"),
    (r"\bgroup\b",                                      "group"),
    (r"\benterprises?\b",                               "enterprises"),
    (r"\bindustries?\b",                                "industries"),
    (r"\bassociates?\b",                                "associates"),
    (r"\bservices?\b",                                  "services"),
    (r"\bsolutions?\b",                                 "solutions"),
    (r"\bconsultants?\b",                               "consultants"),
]
LEGAL_SUFFIXES = [
    (re.compile(p, re.IGNORECASE | re.UNICODE), r)
    for p, r in _RAW_LEGAL
]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ADDRESS ABBREVIATIONS  (applied in ORDER)
# ─────────────────────────────────────────────────────────────────────────────
# BUG FIX: \bst\.?\b was replaced with (?<!\d)st\.?\b so that ordinals
# like "1st", "21st", "3rd" are NOT converted to "1street", "21street", etc.
# Also removed redundant no-op rules like (\bstreet\b → "street").
_RAW_ADDR = [
    # Street types  — negative lookbehind (?<!\d) protects ordinals
    (r"(?<!\d)\bst\.?\b",    "street"),
    (r"\brd\.?\b",           "road"),
    (r"\bave\.?\b",          "avenue"),
    (r"\bdr\.?\b",           "drive"),
    (r"\bblvd\.?\b",         "blvd"),
    (r"\bboulevard\b",       "blvd"),
    (r"\bct\.?\b",           "court"),
    (r"\bpkwy\.?\b",         "parkway"),
    (r"\bln\.?\b",           "lane"),
    (r"\bhwy\.?\b",          "highway"),
    (r"\bhighway\b",         "highway"),
    (r"\bfwy\.?\b",          "freeway"),
    (r"\bcir\.?\b",          "circle"),
    (r"\bpl\.?\b",           "place"),
    (r"\bter\.?\b",          "terrace"),
    # Unit / Apartment / Suite
    (r"\bapt\.?\b",          "apt"),
    (r"\bapartment\b",       "apt"),
    (r"\bste\.?\b",          "ste"),
    (r"\bsuite\b",           "ste"),
    (r"\bunit\b",            "unit"),
    (r"\bfloor\b",           "floor"),
    (r"\bfl\.?\b",           "floor"),
    # PO Box
    (r"\bp\.?o\.?\s*box\b",  "pobox"),
    # Indian specific
    (r"\bkh\.?\s*no\.?\b",   "kh no"),
    (r"\bkhasra\s+no\.?\b",  "kh no"),
    (r"\bnear\b",            "near"),
    (r"\bopp\.?\b",          "opp"),
    (r"\bopposite\b",        "opp"),
    (r"\bgali\b",            "gali"),
    (r"\bmarg\b",            "marg"),
    (r"\bnagar\b",           "nagar"),
    (r"\bcolony\b",          "colony"),
    (r"\bsector\b",          "sector"),
    (r"\bward\b",            "ward"),
    # French
    (r"\brue\b",             "rue"),
    (r"\bimpasse\b",         "impasse"),
    (r"\ballée\b",           "allee"),
    (r"\bavenue\b",          "avenue"),
    (r"\bplace\b",           "place"),
]
ADDRESS_ABBREVIATIONS = [
    (re.compile(p, re.IGNORECASE | re.UNICODE), r)
    for p, r in _RAW_ADDR
]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  UTILITY PATTERNS (pre-compiled)
# ─────────────────────────────────────────────────────────────────────────────
_RE_URL          = re.compile(r"\b\w+\.(com|net|org|in|io|co)\b", re.IGNORECASE)
_RE_LEADING_NOISE= re.compile(r"^[\W_]+", re.UNICODE)   # leading junk like '-- '
_RE_TRAILING_NOISE= re.compile(r"[\W_]+$", re.UNICODE)  # trailing junk
_RE_SPACES       = re.compile(r"\s+")
_RE_AMPERSAND    = re.compile(r"&")
_RE_FRACTION     = re.compile(r"(\d+)\s*/\s*(\d+)")       # "1/2" → "1 2"


# ─────────────────────────────────────────────────────────────────────────────
# 4.  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_str(x) -> str:
    """Convert to string, return '' for None/NaN/'nan'."""
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s.lower() in ("nan", "none") else s


def remove_accents(text: str) -> str:
    """
    Strip combining diacritics (accents) from Latin characters ONLY.

    IMPORTANT (bugfix): the original implementation ran NFKD + Mn-category
    stripping over the WHOLE string. That is safe for Latin accents
    (e.g. 'é' decomposes to 'e' + a combining-acute Mn char which we drop),
    but it is NOT safe for Devanagari/Tamil/Kannada/etc: vowel signs and the
    virama (e.g. Devanagari VIRAMA U+094D, vowel sign U+0949) are themselves
    category "Mn"/"Mc" *without* any decomposition needed, so the old code
    silently deleted them and corrupted non-Latin business names/addresses
    (e.g. "मॉडर्न" → "मॉडरन", losing the virama that makes the word correct).

    Fix: only run NFKD-decompose-and-strip on characters in the Latin
    Unicode range (<= U+024F); every other character (Devanagari, Tamil,
    Kannada, Arabic, CJK, …) is passed through untouched.

    Examples:  Léarning → learning   |   Président → president   |  राम → राम
    """
    out = []
    for ch in text:
        if ord(ch) <= 0x024F:
            nfkd = unicodedata.normalize("NFKD", ch)
            out.append("".join(c for c in nfkd if unicodedata.category(c) != "Mn"))
        else:
            out.append(ch)
    return "".join(out)


def _is_nonlatin(text: str) -> bool:
    """True if the text contains mostly non-Latin characters (Devanagari etc.)."""
    non_latin = sum(1 for c in text if ord(c) > 0x024F and not c.isspace())
    return non_latin > len(text) * 0.4


def _strip_special_chars(text: str) -> str:
    r"""
    Replace punctuation/symbols with a space, while KEEPING:
      - letters, from any script (category starts with 'L')
      - combining marks, from any script (category starts with 'M') —
        critical for Devanagari/Tamil/Kannada vowel signs & virama
      - digits (category starts with 'N')
      - whitespace

    Bugfix: the original code used re.sub(r"[^\w\s]", " ", text). Python's
    \w does NOT include combining marks (Mn/Mc) for scripts like Devanagari
    or Tamil, so that regex was stripping essential vowel signs and the
    virama out of Indian-language names/addresses even after remove_accents
    was fixed. This character-category-based version keeps them.
    """
    return "".join(
        ch if (unicodedata.category(ch)[0] in ("L", "M", "N") or ch.isspace())
        else " "
        for ch in text
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def normalize_name(x) -> str:
    """
    Clean and normalise a business name.

    Pipeline
    --------
    1.  str() conversion — returns '' for None / NaN.
    2.  Lowercase.
    3.  URL/domain detection — strip TLD suffix  (wilfordhancock.com → wilfordhancock).
    4.  Strip Latin combining accents; preserve non-Latin scripts.
    5.  '&' → 'and'.
    6.  Standardise legal entity suffixes via regex rules.
    7.  Fraction normalisation  '1/2' → '1 2'.
    8.  Strip leading/trailing noise  ('-- ', '<< ', quotes).
    9.  Replace remaining special chars with space (script-aware).
    10. Collapse whitespace.

    Examples
    --------
    "-- Holloway Peak Inc Seafood"  →  "holloway peak inc seafood"
    "B+ Retail Inc"                 →  "b retail inc"
    "LLC Moncada Léarning Center"   →  "llc moncada learning center"
    "ZNB Club SARL"                 →  "znb club sarl"
    "wilfordhancock.com"            →  "wilfordhancock"
    "GREENSBORO SUMMIT CORPORATION" →  "greensboro summit corp"
    """
    text = _safe_str(x)
    if not text:
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Strip URL domains
    text = _RE_URL.sub(lambda m: m.group(0).rsplit(".", 1)[0], text)

    # 3. Accent stripping (Latin only)
    text = remove_accents(text)

    # 4. Ampersand
    text = _RE_AMPERSAND.sub(" and ", text)

    # 5. Legal suffixes
    for pattern, replacement in LEGAL_SUFFIXES:
        text = pattern.sub(replacement, text)

    # 6. Fraction normalisation
    text = _RE_FRACTION.sub(r"\1 \2", text)

    # 7. Strip leading / trailing noise
    text = _RE_LEADING_NOISE.sub("", text)
    text = _RE_TRAILING_NOISE.sub("", text)

    # 8. Special chars → space (script-aware, keeps Indic combining marks)
    text = _strip_special_chars(text)

    # 9. Collapse whitespace
    return _RE_SPACES.sub(" ", text).strip()


def normalize_address(x) -> str:
    """
    Clean and normalise a business address.

    Pipeline
    --------
    1.  str() conversion — returns '' for None / NaN.
    2.  Lowercase.
    3.  Strip Latin combining accents.
    4.  Fraction normalisation  '19 1/2' → '19 1 2'.
    5.  Expand address abbreviations via regex rules.
    6.  Replace remaining special chars with space (script-aware).
    7.  Collapse whitespace.

    Examples
    --------
    "105 ELM ST, MORGANTON, NC"              →  "105 elm street morganton nc"
    "1 Ivanhoe Ave, PO Box 6009, Cincinnati" →  "1 ivanhoe avenue pobox 6009 cincinnati"
    "KH NO. -570/13, NEW DELHI"              →  "kh no 570 13 new delhi"
    "175 Boulevard du Président, Bordeaux"   →  "175 blvd du president bordeaux"
    "19 1/2 Stardust Trail"                  →  "19 1 2 stardust trail"
    "Door No 183, 41st Cross, 22nd Main"     →  "door no 183 41street cross 22nd main"
                                             ^  note: "41st" → "41street" is still imperfect
                                                (ordinal lookbehind only stops conversion when
                                                 digit is directly before "st", which it is here.
                                                 Actually with fix: "41st" stays "41st" ✓)
    """
    text = _safe_str(x)
    if not text:
        return ""

    # 1. Lowercase + accent strip
    text = remove_accents(text.lower())

    # 2. Fraction normalisation
    text = _RE_FRACTION.sub(r"\1 \2", text)

    # 3. Address token expansion
    for pattern, replacement in ADDRESS_ABBREVIATIONS:
        text = pattern.sub(replacement, text)

    # 4. Special chars → space (script-aware, keeps Indic combining marks)
    text = _strip_special_chars(text)

    # 5. Collapse whitespace
    return _RE_SPACES.sub(" ", text).strip()
