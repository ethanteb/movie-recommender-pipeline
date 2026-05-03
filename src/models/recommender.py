from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = None
        self.movies = None

    def fit(self, df, text_column: str ="genres"):
        self.movies = df.copy()
        self.movies[text_column] = self.movies[text_column].fillna("")
        self.text_column = text_column
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies[text_column])

    def recommend(self, movie_title: str, top_n: int = 5) -> list:
        matches = self.movies[self.movies["title"].str.lower() == movie_title.lower()]
        
        if matches.empty:
            raise ValueError(f"Movie '{movie_title}' not found")

        idx = matches.index[0]

        similarity_scores = cosine_similarity(
            self.tfidf_matrix[idx],
            self.tfidf_matrix
        ).flatten()

        similarity_scores = list(enumerate(similarity_scores))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        top_movies = similarity_scores[1:top_n+1]
        movie_indices = [i[0] for i in top_movies]

        return self.movies["title"].iloc[movie_indices].tolist()