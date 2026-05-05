class Pipeline:
    #pipeline class bringing together loading, preprocessing, modelling, and evaluation steps
    def __init__(self, loader, preprocessor, model, evaluator):
        self.loader = loader
        self.preprocessor = preprocessor
        self.model = model
        self.evaluator = evaluator

    def run(self):
        df = self.loader.load()
        X_train, X_test, y_train, y_test = self.preprocessor.split(df)
        self.model.train(X_train, y_train)
        predictions = self.model.predict(X_test)
        results = self.evaluator.evaluate(y_test, predictions)
        return results
