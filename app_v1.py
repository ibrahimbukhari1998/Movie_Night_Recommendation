import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Define movies and attributes
movies_metadata = {
    "Parasite": ["Thriller", "Drama", "Dark", "Suspenseful", "Korean", "Modern"],
    "Little Miss Sunshine": ["Comedy", "Drama", "Feel-Good", "Wholesome", "English", "Modern"],
    "Holes": ["Adventure", "Family", "Light", "Engaging", "English", "2000s"],
    "The Man from Earth": ["Sci-Fi", "Drama", "Thought-Provoking", "Minimalist", "English", "2000s"],
    "Midsommar": ["Horror", "Psychological", "Disturbing", "Intense", "English", "Modern"],
    "Life is Beautiful": ["Drama", "War", "Heartwarming", "Tragic", "Italian", "Classic"],
    "Some Like It Hot": ["Comedy", "Romance", "Classic", "Light", "English", "Classic"],
    "The Grand Budapest Hotel": ["Comedy", "Drama", "Quirky", "Aesthetic", "English", "Modern"],
    "Gone Girl": ["Thriller", "Mystery", "Dark", "Twisted", "English", "Modern"],
    "A Silent Voice": ["Anime", "Drama", "Emotional", "Redemptive", "Japanese", "Modern"],
}

# Convert movie attributes to TF-IDF vectors
vectorizer = TfidfVectorizer()
movie_vectors = vectorizer.fit_transform([" ".join(attrs) for attrs in movies_metadata.values()])

# Streamlit UI
def main():
    st.title("🎬 Movie Recommendation Quiz")
    st.write("Answer these fun questions, and we'll find the perfect movie for you!")

    # Questionnaire
    user_answers = []
    user_answers.append(st.selectbox("1. What’s your ideal movie night setup?", ["Emotional", "Suspenseful", "Light & Fun", "Thought-Provoking"]))
    user_answers.append(st.selectbox("2. If your life were a movie, what genre would it be?", ["Comedy", "Thriller", "Drama", "Romance"]))
    user_answers.append(st.selectbox("3. What’s your take on animated movies?", ["Love them!", "Only if they make me cry", "Prefer live-action"]))
    user_answers.append(st.selectbox("4. If you could visit a fictional world, where would you go?", ["Aesthetic", "Thriller", "Romance", "Classic"]))
    user_answers.append(st.selectbox("5. Which plot twist do you love the most?", ["Psychological Thriller", "Mystery", "Drama", "Romance"]))
    user_answers.append(st.selectbox("6. Your dream vacation would be...", ["Classic", "Adventure", "Thriller", "Comedy"]))
    user_answers.append(st.selectbox("7. How do you feel about subtitles?", ["Love foreign films", "Prefer English", "No preference"]))
    user_answers.append(st.selectbox("8. Pick a classic movie quote that speaks to your soul:", ["Classic", "Action", "Romance", "Thriller"]))
    user_answers.append(st.selectbox("9. What kind of endings do you prefer?", ["Feel-Good", "Tragic", "Open-ended", "Shocking"]))
    user_answers.append(st.selectbox("10. What do you do after finishing a movie?", ["Analyze it", "Rewatch scenes", "Move on", "Recommend it"]))
    user_answers.append(st.selectbox("11. Which describes your ideal protagonist?", ["Inspirational", "Thriller", "Comedy", "Romance"]))
    user_answers.append(st.selectbox("12. Your thoughts on horror movies?", ["Love them!", "Only psychological", "No horror", "Horror-comedy"]))
    user_answers.append(st.selectbox("13. What’s your ideal movie soundtrack?", ["Orchestral", "Rock", "Synth", "Romantic"]))
    user_answers.append(st.selectbox("14. Which movie trope do you secretly love?", ["Enemies-to-lovers", "Unreliable narrator", "Misfits unite", "Tragic romance"]))
    user_answers.append(st.selectbox("15. Choose a cinematic color palette:", ["Pastels", "Dark tones", "Bright colors", "Sepia"]))

    if st.button("Get Recommendations!"):
        # Convert user answers to a text format
        user_profile = " ".join(user_answers)
        user_vector = vectorizer.transform([user_profile])

        # Compute cosine similarity
        similarities = cosine_similarity(user_vector, movie_vectors)[0]
        top_movies = sorted(zip(movies_metadata.keys(), similarities), key=lambda x: x[1], reverse=True)[:3]

        st.subheader("🎥 Your Recommended Movies:")
        for movie, score in top_movies:
            st.write(f"**{movie}** (Match Score: {score:.2f})")

if __name__ == "__main__":
    main()
