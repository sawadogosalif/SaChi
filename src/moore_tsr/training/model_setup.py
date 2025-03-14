"""
Note 1 :  
NVIDIA recommande que les tailles des tenseurs (comme l’embedding layer) soient des multiples de **8, 16 ou 32** 
pour maximiser l'utilisation des **Tensor Cores**, qui accélèrent les calculs de multiplication matricielle en virgule flottante.  
C’est documenté ici :  
🔗 [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc)  

Les Tensor Cores sont activés **uniquement** si les dimensions des matrices respectent certains critères :  
- **FP16 et BF16** : multiples de **8**  
- **INT8** : multiples de **16**  
- **FP32** : pas de contrainte mais moins optimisé  

Si la taille du vocabulaire du modèle n'est **pas un multiple de 8**, les calculs basculent en mode non optimisé, ce qui peut entraîner un ralentissement.

Note 2: La library `transformers` a beaucoup changé.

Note 3: Inspired by :
- https://github.com/avidale/
- https://github.com/pollitoconpapas

"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Tuple
from loguru import logger


def fix_tokenizer(tokenizer, new_lang):
    """
    Ajoute un nouveau token de langue au tokenizer et met à jour les mappings d’identifiants.

    - Ajoute le token spécial s'il n'existe pas déjà.
    - Initialise ou met à jour `lang_code_to_id` et `id_to_lang_code` en utilisant `getattr` pour éviter les vérifications répétitives.
    """
    if new_lang not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': [new_lang]})

    tokenizer.lang_code_to_id = getattr(tokenizer, 'lang_code_to_id', {})
    tokenizer.id_to_lang_code = getattr(tokenizer, 'id_to_lang_code', {})

    if new_lang not in tokenizer.lang_code_to_id:
        new_lang_id = tokenizer.convert_tokens_to_ids(new_lang)
        tokenizer.lang_code_to_id[new_lang] = new_lang_id
        tokenizer.id_to_lang_code[new_lang_id] = new_lang

    return tokenizer


def adjust_embeddings(model, tokenizer, new_lang):
    """
    Ajuste les embeddings du modèle après l'ajout d'un nouveau token de langue.

    - Déplace l'embedding du token `<mask>` pour le conserver en dernière position.
    - Initialise l'embedding du nouveau token avec celui d'une langue similaire (ex. `wol_Latn` pour `moor_Latn`).
    """
    added_token_id = tokenizer.convert_tokens_to_ids(new_lang)
    similar_lang_id = tokenizer.convert_tokens_to_ids('wol_Latn')
    embeds = model.model.shared.weight.data

    # Déplacer l'embedding du token `<mask>` vers la position juste après le nouveau token
    embeds[added_token_id + 1] = embeds[added_token_id]

    # Initialiser l'embedding du nouveau token avec celui de la langue similaire
    embeds[added_token_id] = embeds[similar_lang_id]


def setup_model_and_tokenizer(
    model_name: str = "facebook/nllb-200-distilled-600M",
    new_lang_code: str = "moor_Latn",
    device: str = "cuda",
) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    """
    Initialise le modèle et le tokenizer en ajoutant un nouveau token de langue.

    - Charge le tokenizer et ajoute le token de la nouvelle langue.
    - Charge le modèle et ajuste la taille du vocabulaire pour optimiser l'utilisation des Tensor Cores.
    - Applique les ajustements des embeddings.

    Retourne :
    - `model` : Le modèle pré-entraîné avec les embeddings ajustés.
    - `tokenizer` : Le tokenizer mis à jour avec le nouveau token de langue.
    """
    logger.info(f"Model name {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Tokenizer loaded, vocab size = {len(tokenizer)}")

    fix_tokenizer(tokenizer, new_lang_code)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    logger.info("Model loaded.")

    # Ajustement de la taille du vocabulaire (arrondi au multiple de 8 pour Tensor Cores)
    new_vocab_size = (len(tokenizer) + 7) // 8 * 8
    model.resize_token_embeddings(new_vocab_size)
    logger.info(f"New vocab size = {len(tokenizer)}")

    adjust_embeddings(model, tokenizer, new_lang_code)

    model.to(device)

    return model, tokenizer
