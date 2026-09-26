"""Single source of truth for the nine target languages.

Each entry gives the display name, the Wikipedia sub-domain, the indic-nlp
code used for normalisation/tokenisation, the indic_transliteration (sanscript)
scheme name used for romanisation, and the Unicode block used to detect the
target script. Assamese shares the Bengali block.
"""

LANGS = {
    #       name         wiki  indicnlp  sanscript scheme  u_lo      u_hi
    "as": ("Assamese",  "as", "as", "BENGALI",    "ঀ", "৿"),
    "gu": ("Gujarati",  "gu", "gu", "GUJARATI",   "઀", "૿"),
    "kn": ("Kannada",   "kn", "kn", "KANNADA",    "ಀ", "೿"),
    "ml": ("Malayalam", "ml", "ml", "MALAYALAM",  "ഀ", "ൿ"),
    "mr": ("Marathi",   "mr", "mr", "DEVANAGARI", "ऀ", "ॿ"),
    "or": ("Odia",      "or", "or", "ORIYA",      "଀", "୿"),
    "pa": ("Punjabi",   "pa", "pa", "GURMUKHI",   "਀", "੿"),
    "ta": ("Tamil",     "ta", "ta", "TAMIL",      "஀", "௿"),
    "te": ("Telugu",    "te", "te", "TELUGU",     "ఀ", "౿"),
}

ALL_LANGS = list(LANGS)


def lang_info(code: str) -> dict:
    if code not in LANGS:
        raise ValueError(f"Unknown language '{code}'. Choose from {ALL_LANGS}")
    name, wiki, indic, scheme, lo, hi = LANGS[code]
    return dict(code=code, name=name, wiki=wiki, indic=indic,
                scheme=scheme, u_lo=lo, u_hi=hi)


def script_checker(code: str):
    """Return fn(text) -> True if text contains any char of the target script."""
    info = lang_info(code)
    lo, hi = info["u_lo"], info["u_hi"]
    return lambda text: any(lo <= ch <= hi for ch in text)
