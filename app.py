import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
 
import streamlit as st
from src.predict import predict_rating, known_actors, known_directors, known_genres
 
st.set_page_config(page_title="Movie Rating Predictor", page_icon="🎬", layout="centered")
st.title("🎬 Movie Rating Predictor")
st.write(
    "Enter the cast, director, genre(s), and a short synopsis to get a predicted rating."
)
 
actor_options = known_actors()
director_options = known_directors()
genre_options = known_genres()
 
actors_selected = st.multiselect(
    "Actors (pick from known cast, or use 'Other' below to type a name)",
    options=actor_options,
    default=actor_options[:2] if len(actor_options) >= 2 else actor_options,
)
 
extra_actor_text = st.text_input(
    "Additional actor name(s), comma-separated (optional — for names not in the list above)"
)
 
director_selected = st.selectbox(
    "Director",
    options=["-- type a custom name below --"] + director_options,
)
 
custom_director = ""
if director_selected == "-- type a custom name below --":
    custom_director = st.text_input("Custom director name")
 
genres_selected = st.multiselect("Genre(s)", options=genre_options, default=genre_options[:1])
 
synopsis_input = st.text_area(
    "Synopsis",
    placeholder="A love story unfolds aboard a doomed ocean liner...",
    height=150,
)
 
if st.button("Predict Rating", type="primary"):
    all_actors = list(actors_selected)
    if extra_actor_text.strip():
        all_actors += [a.strip() for a in extra_actor_text.split(",") if a.strip()]
 
    final_director = custom_director.strip() if director_selected == "-- type a custom name below --" else director_selected
 
    if not all_actors or not final_director or not genres_selected or not synopsis_input.strip():
        st.warning("Please fill in at least one actor, a director, one genre, and a synopsis.")
    else:
        with st.spinner("Predicting..."):
            result = predict_rating(
                actor_names=all_actors,
                director_name=final_director,
                genres=genres_selected,
                synopsis=synopsis_input,
            )
 
        st.metric("Predicted Rating", f"{result['predicted_rating']} / 10")
 
        if result["unknown_actors"]:
            st.caption(
                f"⚠️ Not seen during training (used dataset average instead): "
                f"{', '.join(result['unknown_actors'])}"
            )
        if result["unknown_director"]:
            st.caption(f"⚠️ Director '{final_director}' not seen during training — used dataset average instead.")
 
st.divider()
st.caption(
    "Model trained on TMDB 5000 data. Actor/director influence is based on their "
    "historical average movie ratings, so unfamiliar names fall back to the dataset average."
)