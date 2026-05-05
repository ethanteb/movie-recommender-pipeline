from src.data.data_loader import DataLoader
from src.preprocessing.preprocessor import Preprocessor
from src.models.recommender import MovieRecommender
from src.pipeline.pipeline import Pipeline

def main():
    loader = DataLoader("data/movies.csv")
    preprocessor = Preprocessor(text_column="genre")
    recommender = MovieRecommender()
    pipeline = Pipeline(loader, preprocessor, recommender)
    movie = "The Dark Knight"
    recommendations = pipeline.run(movie)
    print(f"Recommendations for {movie}:")
    for rec in recommendations:
        print("-", rec)

if __name__ == "__main__":
    main()