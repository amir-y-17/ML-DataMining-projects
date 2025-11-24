import os
import numpy as np
import pandas as pd


def load_data(file_path: str = "Train1/data/dataset2.xlsx") -> np.ndarray:
    """
    Load data from an Excel file.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    df = pd.read_excel(file_path, header=None)

    if not np.issubdtype(df.to_numpy().dtype, np.number):
        raise ValueError("Dataset contains non-numeric values.")

    data = df.to_numpy()
    return data
