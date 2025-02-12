import streamlit as st
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

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
    "A Silent Voice": ["Anime", "Drama", "Emotional", "Redemptive", "Japanese", "Modern"]
}

# Convert metadata to a DataFrame
movies_df = pd.DataFrame(movies_metadata).T.reset_index()
movies_df.columns = ["Movie", "Genre1", "Genre2", "Tone", "Style", "Language", "Era"]

# Encode categorical variables
movies_df_encoded = pd.get_dummies(movies_df.drop(columns=["Movie"]))
labels = movies_df["Movie"]

# Train decision tree
clf = DecisionTreeClassifier()
clf.fit(movies_df_encoded, labels)

# Streamlit UI
def main():
    st.title("🎬 Movie Recommendation Quiz")
    st.write("Answer these fun questions, and we'll find the perfect movie for you!")

    # Questionnaire
    responses = {}
    responses["Genre1"] = st.selectbox("1. Pick a genre:", ["Thriller", "Comedy", "Adventure", "Sci-Fi", "Horror", "Drama", "Romance", "Anime"])
    responses["Tone"] = st.selectbox("2. How do you like your movies?", ["Dark", "Feel-Good", "Light", "Thought-Provoking", "Disturbing", "Heartwarming", "Quirky", "Emotional"])
    responses["Style"] = st.selectbox("3. Choose a movie style:", ["Suspenseful", "Wholesome", "Engaging", "Minimalist", "Intense", "Tragic", "Aesthetic", "Redemptive"])
    responses["Language"] = st.selectbox("4. What language do you prefer?", ["English", "Korean", "Italian", "Japanese"])
    responses["Era"] = st.selectbox("5. Which movie era do you like best?", ["Classic", "2000s", "Modern"])

    if st.button("Get Recommendations!"):
        # Encode user responses
        user_df = pd.DataFrame([responses])
        user_encoded = pd.get_dummies(user_df)
        user_encoded = user_encoded.reindex(columns=movies_df_encoded.columns, fill_value=0)
        
        # Predict top 3 movies
        predictions = clf.predict_proba(user_encoded)[0]
        top_indices = np.argsort(predictions)[-3:][::-1]
        top_movies = labels.iloc[top_indices]
        
        st.subheader("🎥 Your Recommended Movies:")
        for movie in top_movies:
            st.write(f"**{movie}**")

if __name__ == "__main__":
    main()
