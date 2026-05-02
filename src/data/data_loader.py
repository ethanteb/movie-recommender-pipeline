import pandas as pd

class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.filepath)
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()