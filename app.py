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
    "The Rescue": ["Documentary", "Thriller", "Inspirational", "Real-Life", "English", "Modern"],
    "Falling in Love Like in Movies": ["Romance", "Drama", "Heartfelt", "Whimsical", "Indonesian", "Modern"],
    "Barfi": ["Comedy", "Drama", "Heartwarming", "Silent-Romantic", "Hindi", "Modern"],
    "3 Idiots": ["Comedy", "Drama", "Inspirational", "Feel-Good", "Hindi", "Modern"],
    "Haider": ["Drama", "Thriller", "Dark", "Political", "Hindi", "Modern"],
    "Andhadhun": ["Thriller", "Mystery", "Dark", "Twisted", "Hindi", "Modern"],
    "Laapata Ladies": ["Comedy", "Drama", "Light", "Social Commentary", "Hindi", "Modern"],
    "Dangal": ["Sports", "Drama", "Inspirational", "Family", "Hindi", "Modern"],
    "Kal Ho Naa Ho": ["Romance", "Drama", "Emotional", "Feel-Good", "Hindi", "2000s"],
    "Kuch Kuch Hota Hai": ["Romance", "Drama", "Classic", "Wholesome", "Hindi", "Classic"]
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
    user_answers.append(st.selectbox("1. What’s your ideal movie night setup?", ["Dim lights, mystery, and suspense!", "Laughter with popcorn in hand", "A thought-provoking experience", "A heartwarming story with tissues ready"], index=0))
    user_answers.append(st.selectbox("2. If your life were a movie, what genre would it be?", ["A thrilling whodunit", "A goofy feel-good comedy", "A mind-bending drama", "A romantic tearjerker"], index=0))
    user_answers.append(st.selectbox("3. What’s your take on animated movies?", ["Only if they shatter my emotions", "Love me some nostalgia!", "Eh, I prefer realism", "Give me anime or give me nothing"], index=0))
    user_answers.append(st.selectbox("4. If you could visit a fictional world, where would you go?", ["A neon-lit cyberpunk city", "A quirky European town", "A psychological maze of twists", "A fairy-tale love story"], index=0))
    user_answers.append(st.selectbox("5. Which plot twist do you love the most?", ["The killer was inside the house all along!", "The underdog rises to greatness", "The protagonist was the villain!", "They were soulmates in another life"], index=0))
    user_answers.append(st.selectbox("6. Your dream vacation would be...", ["Exploring a haunted mansion", "Backpacking through vibrant cities", "Relaxing in a dreamy countryside", "A fast-paced action adventure"], index=0))
    user_answers.append(st.selectbox("7. How do you feel about subtitles?", ["Foreign films are cinematic gold", "English all the way", "I’ll take dubs if I must", "I enjoy a mix of both"], index=0))
    user_answers.append(st.selectbox("8. Pick a classic movie quote that speaks to your soul:", ["I see dead people", "Life is like a box of chocolates", "Here's looking at you, kid", "May the Force be with you"], index=0))
    user_answers.append(st.selectbox("9. What kind of endings do you prefer?", ["Mind-blowing cliffhangers", "Happily ever after", "Bittersweet but meaningful", "A good old-fashioned twist"], index=0))
    user_answers.append(st.selectbox("10. What do you do after finishing a movie?", ["Scour the internet for theories", "Immediately recommend it to friends", "Cry and relive it in my head", "Start watching the next one"], index=0))
    user_answers.append(st.selectbox("11. Which describes your ideal protagonist?", ["A misunderstood genius", "A lovable misfit", "A determined underdog", "A hopeless romantic"], index=0))
    user_answers.append(st.selectbox("12. Your thoughts on horror movies?", ["Give me nightmares!", "Only psychological ones", "No thanks!", "Only if they’re funny"], index=0))
    user_answers.append(st.selectbox("13. What’s your ideal movie soundtrack?", ["Epic orchestral pieces", "80s synth vibes", "Melancholic piano tunes", "Catchy indie folk"], index=0))
    user_answers.append(st.selectbox("14. Which movie trope do you secretly love?", ["The villain with a tragic backstory", "The unlikely group of heroes", "The slow-burn romance", "The shocking identity reveal"], index=0))
    user_answers.append(st.selectbox("15. Choose a cinematic color palette:", ["Moody blues and grays", "Bright technicolor", "Pastel dreamscapes", "Golden vintage tones"], index=0))


    if st.button("Get Recommendations!"):
        # Convert user answers to a text format
        user_profile = " ".join(user_answers)
        user_vector = vectorizer.transform([user_profile])

        # Compute cosine similarity
        similarities = cosine_similarity(user_vector, movie_vectors)[0]
        top_movies = sorted(zip(movies_metadata.keys(), similarities), key=lambda x: x[1], reverse=True)[:10]

        st.subheader("🎥 Your Recommended Movies:")
        for movie, score in top_movies:
            st.write(f"**{movie}** (Match Score: {score:.2f})")

if __name__ == "__main__":
    main()
