import pandas as pd
from torch.utils.data import Dataset


class NameDataset(Dataset):
    def __init__(self, df: pd.DataFrame, col_name: str):
        self.data_frame = df[col_name]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        return self.data_frame.iloc[index]
