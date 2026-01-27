from pathlib import Path

from data_loader import load_movies
from recommender import ContentBasedRecommender


def main():
    try:
        project_root = Path(__file__).parent
        csv_path = project_root / "data" / "movies.csv"
        print("CSV path:", csv_path)

        movies_df = load_movies(str(csv_path))
        print("Loaded movies:")
        print(movies_df[["id", "title", "genres"]])

        model = ContentBasedRecommender()
        model.fit(movies_df)
        print("\nModel fitted on movies dataset.\n")

        while True:
            user_input = input(
                "Type name of the movie you like ('exit' to quit): "
            ).strip()

            if user_input.lower() == "exit":
                print("exit the program")
                break

            if not user_input:
                continue

            try:
                recommendations = model.recommend_by_title(user_input, top_n=5)
            except ValueError as e:
                print(e)
                continue

            print(f"\nRecommended movies similar to '{user_input}':")
            for title, score in recommendations:
                print(f"  - {title}  (similarity = {score:.3f})")
            print()

    except Exception as e:
        print("\n=== ERROR OCCURRED ===")
        print(repr(e))
        input("\nEnter")


if __name__ == "__main__":
    main()
