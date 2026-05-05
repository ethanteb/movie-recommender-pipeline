class Preprocessor:
    def __init__(self, target_column: str):
        self.target_column = target_column

    def transform(self, df):
        df[self.target_column] = df[self.target_column].fillna("")
        return df