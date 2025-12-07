import pandas as pd
import numpy as np
import re

def convert_to_numeric(col: pd.Series) -> bool:
    col.replace('?', np.nan, inplace=True)  # replacing empty strings with NaN
    col.dropna(inplace=True)  # dropping NaN values
    for val in col.unique(): # getting unique values of the column and chcking if this can be cconverted to float
        try:
            float(val)
        except ValueError:
            return False
    return True

def is_numeric_column(col: pd.Series) -> bool:
    return bool(re.search(r"[A-Za-z]", str(col))) 