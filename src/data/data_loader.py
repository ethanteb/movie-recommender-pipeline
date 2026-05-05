import pandas as pd

class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load(self) -> pd.DataFrame:
        try:
            data = pd.read_csv(self.filepath)
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()