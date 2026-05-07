import pandas as pd

class Preprocessor:
    def __init__(self, weights: dict = None):
        self.weights = weights or {
            "genre": 3,
            "directors": 2,
            "writers": 1,
            "cast": 2,
            "studio_name": 1
        }


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        parts = []
        for col, weight in self.weights.items():
            parts.append((df[col] + " ") * weight)

        df["text_features"] = (
            pd.concat(parts, axis=1)
            .sum(axis=1)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        df["normalized_ratings"] = (df["tomatometer_rating"] + df["audience_rating"]) / 200
        return df