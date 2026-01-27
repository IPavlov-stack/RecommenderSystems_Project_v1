import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_metrics(df_metrics: pd.DataFrame):
    """Визуализира Precision@N, Recall@N и Avg Cosine Similarity."""
    plt.figure(figsize=(10, 6))

    plt.plot(df_metrics["N"], df_metrics["Precision@N"], marker='o', label="Precision@N")
    plt.plot(df_metrics["N"], df_metrics["Recall@N"], marker='s', label="Recall@N")
    plt.plot(df_metrics["N"], df_metrics["Avg Cosine Similarity"], marker='^', label="Avg Cosine Similarity")

    plt.xlabel("N (Top-N Recommendations)")
    plt.ylabel("Metric Value")
    plt.title("Evaluation Metrics for Content-Based Recommender")
    plt.xticks(df_metrics["N"])
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    metrics_csv = Path(__file__).parent / "metrics.csv"
    
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")
    
    df_metrics = pd.read_csv(metrics_csv)
    plot_metrics(df_metrics)
