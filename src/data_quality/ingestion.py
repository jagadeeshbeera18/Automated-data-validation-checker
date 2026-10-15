from pathlib import Path
import pandas as pd
from typing import Union


def load_csv(file_path:Union[str,Path])->pd.DataFrame:
    path=Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found:{path}")
    if not path.is_file():
        raise ValueError(f"Expected a file received:{path}")
    if path.suffix.lower()!= ".csv":
        raise ValueError(f"Expected as csv file, received:{path.suffix}")
    return pd.read_csv(path)
