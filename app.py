import streamlit as st
from sentence_transformers import SentenceTransformer, util
import numpy as np

def movie_recommendation():
    movies = [
        "Parasite", "Little Miss Sunshine", "Holes", "The Man from Earth", "Midsommar",
        "Life is Beautiful", "Some Like It Hot", "The Grand Budapest Hotel", "Gone Girl", "A Silent Voice",
        "The Rescue", "Falling in Love Like in Movies", "Barfi", "3 Idiots", "Haider",
        "Andhadhun", "Laapata Ladies", "Dangal", "Kal Ho Naa Ho", "Kuch Kuch Hota Hai"
    ]
    
    movie_descriptions = {
        "Parasite": "Dark comedy thriller about social class struggle.",
        "Little Miss Sunshine": "A heartfelt road trip comedy-drama about family.",
        "Holes": "A mystery adventure with themes of fate and justice.",
        "The Man from Earth": "A philosophical sci-fi film about immortality.",
        "Midsommar": "A psychological horror set in a Swedish cult.",
        "Life is Beautiful": "A poignant story about love and hope during the Holocaust.",
        "Some Like It Hot": "A classic screwball comedy with mistaken identities.",
        "The Grand Budapest Hotel": "A whimsical tale of a hotel concierge’s adventures.",
        "Gone Girl": "A psychological thriller with twists and mind games.",
        "A Silent Voice": "An emotional anime drama about bullying and redemption.",
        "The Rescue": "A gripping documentary about the Thai cave rescue.",
        "Falling in Love Like in Movies": "A romantic drama about love and expectations.",
        "Barfi": "A touching Bollywood love story with a unique protagonist.",
        "3 Idiots": "A Bollywood comedy-drama about education and friendship.",
        "Haider": "A Shakespearean tragedy set in Kashmir.",
        "Andhadhun": "A suspenseful thriller about a blind pianist and a crime.",
        "Laapata Ladies": "A social drama about women’s empowerment.",
        "Dangal": "A sports biopic about women’s wrestling in India.",
        "Kal Ho Naa Ho": "A romantic Bollywood film with a tragic twist.",
        "Kuch Kuch Hota Hai": "A Bollywood romance about friendship and love."
    }
    
    questions = [
        "Do you prefer comedy over drama?",
        "Do you enjoy psychological thrillers?",
        "Are you a fan of romance movies?",
        "Do you like movies based on real events?",
        "Do you prefer animated movies?",
        "Do you like fast-paced action over slow-burn storytelling?",
        "Do you prefer movies with deep philosophical themes?",
        "Do you enjoy horror movies?",
        "Do you like Bollywood films?",
        "Are you interested in documentaries?"
    ]
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    user_profile = []
    
    st.title("Movie Recommendation System")
    st.write("Answer the following questions to get personalized movie recommendations:")
    
    for question in questions:
        answer = st.radio(question, ('Yes', 'No'), index=1)
        if answer == 'Yes':
            user_profile.append(question)
    
    if st.button("Get Recommendations"):
        if not user_profile:
            st.write("You didn't select any preferences. Showing all movies.")
            st.write(movies)
        else:
            user_embedding = model.encode(" ".join(user_profile), convert_to_tensor=True)
            movie_scores = []
            
            for movie, desc in movie_descriptions.items():
                movie_embedding = model.encode(desc, convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(user_embedding, movie_embedding).item()
                movie_scores.append((movie, similarity))
            
            ranked_movies = sorted(movie_scores, key=lambda x: x[1], reverse=True)
            
            st.write("### Recommended Movies in Order:")
            for rank, (movie, score) in enumerate(ranked_movies, 1):
                st.write(f"{rank}. {movie} (Score: {score:.2f})")

if __name__ == "__main__":
    movie_recommendation()
