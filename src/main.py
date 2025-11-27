# src/main.py

from data_loader import load_data
from preprocessing import preprocess
from model import train_model, evaluate_model
from utils import save_results


def main():
    print("🔹 Loading data...")
    data = load_data()

    print("🔹 Preprocessing data...")
    processed_data = preprocess(data)

    print("🔹 Training model...")
    model, metrics = train_model(processed_data)

    print("🔹 Evaluating model...")
    eval_results = evaluate_model(model, processed_data)

    print("🔹 Saving results...")
    save_results(model, metrics, eval_results)

    print("🎉 Pipeline finished successfully!")


if __name__ == "__main__":
    main()
