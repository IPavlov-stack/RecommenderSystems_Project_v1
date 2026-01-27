import numpy as np
import pandas as pd
from pathlib import Path

from data_loader import load_movies
from recommender import ContentBasedRecommender


def genres_of(title: str, df: pd.DataFrame) -> set:
    """Returns a set of genres for a given title, safe against missing or NaN genres."""
    row = df[df["title"].str.lower() == title.lower()]
    if row.empty:
        return set()
    genres_str = str(row.iloc[0].get("genres", ""))
    return set(g.strip() for g in genres_str.split() if g.strip())


def evaluate_model(csv_path: Path, N_values=(5, 10, 25)) -> pd.DataFrame:
    """Evaluates ContentBasedRecommender on the given CSV file."""
    df = load_movies(str(csv_path))
    model = ContentBasedRecommender()
    model.fit(df)

    # Precompute genres for all titles
    title_to_genres = {title: genres_of(title, df) for title in df["title"]}

    results = []

    for N in N_values:
        precisions = []
        recalls = []
        similarities = []

        for title in df["title"]:
            try:
                recs = model.recommend_by_title(title, top_n=N)
            except ValueError:
                continue  # skip titles not found

            query_genres = title_to_genres.get(title, set())
            if not query_genres:
                continue

            # Total relevant movies
            relevant_total = sum(
                1 for other_title, genres in title_to_genres.items()
                if other_title != title and genres & query_genres
            )
            if relevant_total == 0:
                continue

            relevant_found = 0
            for rec_title, sim in recs:
                similarities.append(sim)
                rec_genres = title_to_genres.get(rec_title, set())
                if rec_genres & query_genres:
                    relevant_found += 1

            precisions.append(relevant_found / N)
            recalls.append(relevant_found / relevant_total)

        avg_precision = np.mean(precisions) if precisions else 0.0
        avg_recall = np.mean(recalls) if recalls else 0.0
        avg_similarity = np.mean(similarities) if similarities else 0.0

        results.append({
            "N": N,
            "Precision@N": avg_precision,
            "Recall@N": avg_recall,
            "Avg Cosine Similarity": avg_similarity
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    csv_path = Path(__file__).parent / "data" / "movies.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at expected location: {csv_path}")

    df_metrics = evaluate_model(csv_path, N_values=(5, 10, 25))
    print(df_metrics)
metrics_csv_path = Path(__file__).parent / "metrics.csv"
df_metrics.to_csv(metrics_csv_path, index=False)
print(f"Metrics saved to {metrics_csv_path}")
