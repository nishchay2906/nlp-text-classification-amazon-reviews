import argparse
import csv
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline


def load_reviews(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return [row["Sentence"] for row in rows], [int(row["Polarity"]) for row in rows]


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate an Amazon review sentiment baseline.")
    root = Path(__file__).resolve().parent
    parser.add_argument("--train", type=Path, default=root / "sentiment_train.csv")
    parser.add_argument("--test", type=Path, default=root / "sentiment_test.csv")
    args = parser.parse_args()

    train_text, train_labels = load_reviews(args.train)
    test_text, test_labels = load_reviews(args.test)
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression())
    model.fit(train_text, train_labels)
    print(classification_report(test_labels, model.predict(test_text), digits=3))


if __name__ == "__main__":
    main()
