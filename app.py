import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Define movies and attributes (same as before)
movies_metadata = {
    "Parasite": ["Thriller", "Drama", "Dark", "Suspenseful", "Korean", "Modern"],
    "Little Miss Sunshine": ["Comedy", "Drama", "Feel-Good", "Wholesome", "English", "Modern"],
    "Holes": ["Adventure", "Family", "Light", "Engaging", "English", "2000s"],
    # ... (remaining movie data)
}

# Convert metadata to a DataFrame
movies_df = pd.DataFrame(movies_metadata).T.reset_index()
movies_df.columns = ["Movie", "Genre1", "Genre2", "Tone", "Style", "Language", "Era"]

# Encode categorical variables
movies_df_encoded = pd.get_dummies(movies_df.drop(columns=["Movie"]))
labels = movies_df["Movie"]

# Use RandomForest for better performance
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(movies_df_encoded, labels)

# Streamlit UI
def main():
    st.title("🎬 Movie Recommendation Quiz")
    st.write("Answer these fun questions, and we'll find the perfect movie for you!")

    # Questionnaire
    responses = {}
    responses["Genre1"] = st.multiselect("1. Pick your favorite genres:", ["Thriller", "Comedy", "Adventure", "Sci-Fi", "Horror", "Drama", "Romance", "Anime"])
    responses["Tone"] = st.selectbox("2. How do you like your movies?", ["Dark", "Feel-Good", "Light", "Thought-Provoking", "Disturbing", "Heartwarming", "Quirky", "Emotional"])
    responses["Style"] = st.selectbox("3. Choose a movie style:", ["Suspenseful", "Wholesome", "Engaging", "Minimalist", "Intense", "Tragic", "Aesthetic", "Redemptive"])
    responses["Language"] = st.selectbox("4. What language do you prefer?", ["English", "Korean", "Italian", "Japanese", "Hindi", "Indonesian"])
    responses["Era"] = st.selectbox("5. Which movie era do you like best?", ["Classic", "2000s", "Modern"])

    # Add more questions
    responses["CharacterType"] = st.selectbox("6. What type of characters do you enjoy?", ["Strong", "Vulnerable", "Quirky", "Relatable", "Mysterious", "Funny"])
    responses["PlotTwist"] = st.selectbox("7. How much do you enjoy plot twists?", ["None", "Small surprises", "Medium twists", "Big twists", "I love the unexpected!"])
    responses["Length"] = st.selectbox("8. What length do you prefer?", ["Short (less than 90 mins)", "Medium (90-120 mins)", "Long (over 120 mins)"])
    responses["Setting"] = st.selectbox("9. Which setting do you enjoy the most?", ["Urban", "Nature", "Space", "Historical", "Fantasy", "Small Town"])
    responses["Mood"] = st.selectbox("10. What mood are you in today?", ["Adventurous", "Romantic", "Thoughtful", "Relaxed", "Energetic", "Curious"])

    if st.button("Get Recommendations!"):
        # Encode user responses
        user_df = pd.DataFrame([responses])
        user_encoded = pd.get_dummies(user_df)

        # Reindex to match the movie dataset
        user_encoded = user_encoded.reindex(columns=movies_df_encoded.columns, fill_value=0)

        # Predict top 3 movies based on user preferences
        predictions = clf.predict_proba(user_encoded)[0]
        top_indices = np.argsort(predictions)[-5:][::-1]
        top_movies = labels.iloc[top_indices]

        # Display Recommendations
        st.subheader("🎥 Your Recommended Movies:")
        for movie in top_movies:
            st.write(f"**{movie}**")

        # Show Movie Poster (for fun)
        movie_posters = {
            "Parasite": "https://m.media-amazon.com/images/M/MV5BYjk1Y2U4MjQtY2ZiNS00OWQyLWI3MmYtZWUwNmRjYWRiNWNhXkEyXkFqcGc@._V1_.jpg",
            # "Little Miss Sunshine": "https://link_to_poster_LittleMissSunshine.jpg",
            # # ... add links to other movie posters
        }
        for movie in top_movies:
            st.image(movie_posters.get(movie, "https://i.ytimg.com/vi/2oB9emtxdrk/hqdefault.jpg"), caption=movie, width=200)

if __name__ == "__main__":
    main()
