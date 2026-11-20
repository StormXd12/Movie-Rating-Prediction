import json
import os
import joblib
import numpy as np
import scipy.sparse as sp
from xgboost import XGBRegressor
 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
def _load_artifacts():
    model = XGBRegressor()
    model.load_model(os.path.join(MODELS_DIR, "movie_rating_model.json"))
 
    tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    mlb = joblib.load(os.path.join(MODELS_DIR, "genre_binarizer.pkl"))
 
    with open(os.path.join(MODELS_DIR, "actor_rating_lookup.json")) as f:
        actor_lookup = json.load(f)
    with open(os.path.join(MODELS_DIR, "director_rating_lookup.json")) as f:
        director_lookup = json.load(f)
    with open(os.path.join(MODELS_DIR, "global_avg_rating.json")) as f:
        global_avg = json.load(f)["global_avg_rating"]
 
    return model, tfidf, mlb, actor_lookup, director_lookup, global_avg
model, tfidf, mlb, actor_lookup, director_lookup, global_avg = _load_artifacts()
 
 
def known_actors() -> list[str]:
    """Return the sorted list of actor names seen during training (for dashboard autocomplete)."""
    return sorted(actor_lookup.keys())
 
 
def known_directors() -> list[str]:
    """Return the sorted list of director names seen during training (for dashboard autocomplete)."""
    return sorted(director_lookup.keys())
 
 
def known_genres() -> list[str]:
    """Return the list of genre labels the model was trained on."""
    return list(mlb.classes_)
 
 
def predict_rating(actor_names: list[str], director_name: str, genres: list[str], synopsis: str):
    """
    Returns a dict:
        {
            "predicted_rating": float,
            "unknown_actors": [...],   # actor names not seen during training
            "unknown_director": bool,  # True if director not seen during training
        }
    """
    # --- Actor feature ---
    known = [a for a in actor_names if a in actor_lookup]
    unknown_actors = [a for a in actor_names if a not in actor_lookup]
    actor_scores = [actor_lookup[a] for a in known]
    actor_avg = np.mean(actor_scores) if actor_scores else global_avg
 
    # --- Director feature ---
    unknown_director = director_name not in director_lookup
    director_avg = director_lookup.get(director_name, global_avg)
 
    # --- Genre feature (must use the SAME fitted binarizer as training) ---
    genre_vector = mlb.transform([genres])  # shape (1, n_genres)
 
    # --- Synopsis feature ---
    synopsis_vector = tfidf.transform([synopsis])  # sparse (1, n_tfidf_features)
 
    # --- Assemble in the SAME column order used during training ---
    structured = np.array([[actor_avg, director_avg]])
    structured_full = np.hstack([structured, genre_vector])
    X_input = sp.hstack([sp.csr_matrix(structured_full), synopsis_vector])
 
    raw_prediction = model.predict(X_input)[0]
    predicted_rating = round(float(np.clip(raw_prediction, 0, 10)), 2)
 
    return {
        "predicted_rating": predicted_rating,
        "unknown_actors": unknown_actors,
        "unknown_director": unknown_director,
    }
 
 
if __name__ == "__main__":
    # Quick manual test — run with: python src/predict.py
    result = predict_rating(
        actor_names=["Leonardo DiCaprio", "Kate Winslet"],
        director_name="James Cameron",
        genres=["Drama", "Romance"],
        synopsis="A love story unfolds aboard a doomed ocean liner.",
    )
    print(result) 