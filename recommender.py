import numpy as np
import pandas as pd
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = None
        self.movies_df = None

    def fit(self, movies_df: pd.DataFrame):
        if "text" not in movies_df.columns:
            raise ValueError("DataFrame must contain 'text' column")

        self.movies_df = movies_df.reset_index(drop=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.movies_df["text"].tolist()
        )

    def recommend_by_title(self, title: str, top_n: int = 5) -> List[Tuple[str, float]]:
        return self.recommend_by_titles([title], top_n)

    def recommend_by_titles(self, titles: List[str], top_n: int = 5):
        if self.movies_df is None or self.tfidf_matrix is None:
            raise RuntimeError("Model is not fitted.")

        indices = []
        for t in titles:
            rows = self.movies_df[self.movies_df["title"] == t]
            if rows.empty:
                raise ValueError(f"Movie '{t}' not found.")
            indices.append(rows.index[0])

        profile = self.tfidf_matrix[indices].mean(axis=0)
        profile = np.asarray(profile)

        # Correct cosine similarity call
        similarities = cosine_similarity(profile, self.tfidf_matrix).ravel()

        for idx in indices:
            similarities[idx] = -1.0

        ranked = similarities.argsort()[::-1]

        results = []
        for i in ranked:
            if similarities[i] < 0:
                continue
            results.append(
                (self.movies_df.loc[i, "title"], float(similarities[i]))
            )
            if len(results) >= top_n:
                break

        return results
