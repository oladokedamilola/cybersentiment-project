import joblib
import numpy as np
from pathlib import Path


MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'classifier.joblib'
VECT_PATH = Path(__file__).resolve().parents[1] / 'models' / 'vectorizer.joblib'


# NOTE: Put your trained classifier and vectorizer in `models/` folder or change paths.


_model = None
_vectorizer = None




def load_model():
    global _model, _vectorizer
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _vectorizer is None:
        _vectorizer = joblib.load(VECT_PATH)
    return _model, _vectorizer




def predict(texts):
    model, vect = load_model()
    X = vect.transform(texts)
    probs = model.predict_proba(X)
    classes = model.classes_
    preds = model.predict(X)
    # map to (label, score) where score is probability of the predicted label
    results = []
    for p, pred in zip(probs, preds):
        label_index = list(classes).index(pred)
        score = float(p[label_index])
        results.append((pred, score))
    return results