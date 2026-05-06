class Pipeline:
    def __init__(self, loader, preprocessor, model):
        self.loader = loader
        self.preprocessor = preprocessor
        self.model = model

    def run(self, movie_title: str):
        df = self.loader.load()
        df = self.preprocessor.transform(df)
        self.model.fit(df, text_column="text_features")
        recommendations = self.model.recommend(movie_title, top_n=5)
        return recommendations