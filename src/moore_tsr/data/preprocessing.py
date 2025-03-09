import re
import unicodedata
import pandas as pd
import swifter
from sacremoses import MosesPunctNormalizer

mpn = MosesPunctNormalizer(lang="en")

# Précompiler les regex des substitutions pour éviter de les recréer à chaque appel
mpn.substitutions = [(re.compile(r), sub) for r, sub in mpn.substitutions]

# Fonction optimisée de nettoyage
def preprocess_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""

    text = mpn.normalize(text)
    text = re.sub(r"[\x00-\x1F\x7F-\x9F]", " ", text)
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()
