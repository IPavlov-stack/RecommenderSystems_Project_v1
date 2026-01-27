import os
import pandas as pd


def load_movies(csv_path: str) -> pd.DataFrame:
    """
    Loading CSV file with movies/series and return pandas DataFrame.
    add column 'text' = title + description + genres,
    for TF-IDF.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # --    za vseki sluchai
    for col in ["title", "description", "genres"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
        else:
            raise ValueError(f"Missing required column '{col}' in CSV")
    # --    
    
    df["text"] = (
    df["description"].fillna("") + " " +
    df["genres"].fillna("")
                 )

    return df

