import pandas as pd

class Preprocessor:
    def __init__(self):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_combine = [
            "genre",
            "directors",
            "writers",
            "cast",
            "studio_name"
        ]

        df = df.copy()

        for col in cols_to_combine:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("").astype(str)

        df["text_features"] = (
            df[cols_to_combine]
            .agg(" ".join, axis=1)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        return df