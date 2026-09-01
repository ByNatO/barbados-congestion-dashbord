import pandas as pd
import numpy as np

# Ordre exact des caractéristiques attendues par le modèle
FEATURE_ORDER = [
    'signaling', 'day_week', 'hour',
    'mean_r', 'mean_g', 'mean_b',
    'std_r', 'std_g', 'std_b'
]

# Mapping des classes (entiers -> noms compréhensibles)
CLASS_NAMES = {
    0: "free flowing",
    1: "heavy delay",
    2: "light delay",
    3: "moderate delay"
}

def predict_with_proba(model, row):
    """
    Prend une ligne (Series pandas) et retourne la classe prédite en langage clair
    ainsi qu'un dictionnaire des probabilités pour chaque classe.
    """
    X = pd.DataFrame([row[FEATURE_ORDER]], columns=FEATURE_ORDER)

    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0]
        classes = model.classes_  # par ex. [0, 1, 2, 3]

        # Conversion des classes en noms explicites
        mapped_classes = [CLASS_NAMES.get(c, str(c)) for c in classes]
        proba_dict = dict(zip(mapped_classes, proba))

        # Classe prédite
        pred_int = classes[np.argmax(proba)]
        pred_class = CLASS_NAMES.get(pred_int, str(pred_int))

        return pred_class, proba_dict
    else:
        # Fallback si le modèle ne dispose pas de predict_proba
        pred = model.predict(X)[0]
        pred_class = CLASS_NAMES.get(pred, str(pred))
        return pred_class, {}