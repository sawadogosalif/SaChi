from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Tuple
from loguru import logger
la library transformers a beaucoup evolué dernièrement.

"""Note 1 : You are resizing the embedding layer without providing a `pad_to_multiple_of` parameter. This means that the new embedding dimension will be 256205. This might induce some performance reduction as *Tensor Cores* will not be available. For more details about this, or help on choosing the correct value for resizing, refer to this guide: https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
Le message d’avertissement vient de **Hugging Face Transformers** et fait référence aux exigences des **Tensor Cores** de NVIDIA.  NVIDIA recommande que les tailles des tenseurs (comme l’embedding layer) soient des multiples de **8, 16 ou 32** pour maximiser l'utilisation des **Tensor Cores**, qui accélèrent les calculs de multiplication matricielle en virgule flottante. C’est documenté ici :  
🔗 [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc)

Note 2: la library transformers a beaucoup changé
"""





                                           
### Pourquoi ?
Les Tensor Cores sont activés **uniquement** si les dimensions des matrices respectent certains critères :
- **FP16 et BF16** : multiples de **8**
- **INT8** : multiples de **16**
- **FP32** : pas de contrainte mais moins optimisé  

Si la taille du vocabulaire du modèle n'est **pas un multiple de 8**, les calculs basculent en mode non optimisé, ce qui peut entraîner un ralentissement.

### Dois-tu corriger ça ?
Non, ce n’est pas **obligatoire**, mais si tu veux éviter une perte de performance, tu peux forcer l’arrondi en ajoutant :  

```python

```

Ce n’est pas une erreur bloquante, mais une **recommandation de performance**. 🚀


def setup_model_and_tokenizer(
    model_name: str = "facebook/nllb-200-distilled-600M",
    new_lang_code: str = "moore_open",
    device: str = "cuda",
) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    """
    Initialise le modèle et le tokenizer, et ajoute un nouveau token de langue.

    Args:
        model_name (str): Nom du modèle pré-entraîné.
        new_lang_code (str): Code de la nouvelle langue à ajouter.
        device (str): Device à utiliser ("cuda" ou "cpu").

    Returns:
        Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]: Modèle et tokenizer configurés.
    """
    # Chargement du tokenizer
    logger.info(f"Model name {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"tokenize loaded and leen is {len(tokenizer)}.")

    # Ajout du nouveau token de langue
    tokenizer.add_special_tokens({"additional_special_tokens": [new_lang_code]})
    
    # Chargement du modèle
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    logger.info(f"Model loaded.")
    
    new_vocab_size = (len(tokenizer) + 7) // 8 * 8  # Arrondi au multiple de 8 : Voir Note 1
    model.resize_token_embeddings(new_vocab_size)
    logger.info(f"new token length {len(tokenizer)}")

    model.to(device)


    return model, tokenizer
