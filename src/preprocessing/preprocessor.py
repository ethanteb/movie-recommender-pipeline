import pandas as pd
from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self, target_column: str):
        self.target_column = target_column

    def split(self, df: pd.DataFrame):
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        return train_test_split(X, y, test_size=0.2, random_state=42)