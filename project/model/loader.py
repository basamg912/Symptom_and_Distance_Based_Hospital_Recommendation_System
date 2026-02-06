
from sentence_transformers import SentenceTransformer

_model = None

def get_model(model_name='jhgan/ko-sroberta-sts'):
    global _model
    if _model is None:
        print(f"Loading model: {model_name}...")
        _model = SentenceTransformer(model_name)
    return _model
