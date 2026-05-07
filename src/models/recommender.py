from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class MovieRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            sublinear_tf=True
        ) 
        self.tfidf_matrix = None
        self.movies = None
        self.text_column = None

    def fit(self, df, text_column: str = "text_features"):
        self.movies = df.copy()
        self.text_column = text_column

        self.movies[self.text_column] = self.movies[self.text_column].fillna("")

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.movies[self.text_column]
        )

        return self

    def recommend(self, movie_title: str, top_n: int = 5) -> list:
        matches = self.movies[
            self.movies["movie_title"].str.lower() == movie_title.lower()
        ]

        if matches.empty:
            raise ValueError(f"Movie '{movie_title}' not found")

        idx = matches.index[0]

        similarity_scores = cosine_similarity(
            self.tfidf_matrix[idx],
            self.tfidf_matrix
        ).flatten()

        similarity_scores = list(enumerate(similarity_scores))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        scores = np.array([score for _, score in similarity_scores])

        overall_score = (
            self.movies["normalized_ratings"] * 0.25 + scores * 0.75
        )

        top_movies = overall_score.sort_values(ascending=False).iloc[1:top_n + 1]

        results = [
            (self.movies["movie_title"].iloc[i], score)
            for i, score in top_movies.items()
        ]

        return results