from src.data.data_loader import DataLoader
from src.preprocessing.preprocessor import Preprocessor
from src.models.recommender import MovieRecommender
from src.pipeline.pipeline import Pipeline

def main():
    loader = DataLoader("data/simple_movie_data.csv")
    preprocessor = Preprocessor()
    recommender = MovieRecommender()
    pipeline = Pipeline(loader, preprocessor, recommender)

    movie_title = input("Enter a movie title: ")

    recommendations = pipeline.run(movie_title)

    print(f"\nTop recommendations for '{movie_title}':\n")
    for title, score in recommendations:
        print(f"- {title} (score: {score:.3f})")


if __name__ == "__main__":
    main()