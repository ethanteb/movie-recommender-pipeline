from sklearn.metrics import accuracy_score

class Evaluator:
    #class to evaluate the performance of the model using accuracy score metric
    def evaluate(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict:
        accuracy = accuracy_score(y_true, y_pred)
        return {"accuracy": accuracy}