import re
import sys
import typing as tp
import unicodedata
from sacremoses import MosesPunctNormalizer

def get_non_printing_char_replacer(replace_by: str = " ") -> tp.Callable[[str], str]:
    """
    Crée une fonction qui remplace les caractères non imprimables.
    Adapté du dépôt Stopes de l'équipe NLLB.
    """
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }

    def replace_non_printing_char(line) -> str:
        return line.translate(non_printable_map)

    return replace_non_printing_char

def clean_and_normalize(text: str) -> str:
    """
    Nettoie et normalise le texte en :
    1. Normalisant la ponctuation
    2. Supprimant les caractères non imprimables
    3. Normalisant les formes Unicode
    """
    # Normalisation de la ponctuation
    mpn = MosesPunctNormalizer(lang="en")
    mpn.substitutions = [(re.compile(r), sub) for r, sub in mpn.substitutions]
    
    # Suppression des caractères non imprimables
    replace_nonprint = get_non_printing_char_replacer(" ")
    
    # Nettoyage et normalisation
    clean_text = mpn.normalize(text)
    clean_text = replace_nonprint(clean_text)
    clean_text = unicodedata.normalize("NFKC", clean_text)
    
    return clean_text

def preprocess_text(text: str, lang: str = "fr") -> str:
    """
    Prétraite le texte pour l'entraînement du modèle.
    Args:
        text (str): Texte à prétraiter
        lang (str): Code de langue pour la normalisation
    Returns:
        str: Texte prétraité
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Nettoyage de base
    text = clean_and_normalize(text)
    
    # Suppression des espaces multiples
    text = re.sub(r"\s+", " ", text).strip()
    
    return text