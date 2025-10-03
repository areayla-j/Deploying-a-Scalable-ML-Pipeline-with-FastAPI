import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    train_model,
    inference,
    compute_model_metrics,
    save_model,
    performance_on_categorical_slice,
)

def main():
    data = pd.read_csv("data/census.csv")

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    label = "salary"

    train, test = train_test_split(
        data, test_size=0.20, random_state=42, stratify=data[label]
    )

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label=label, training=True
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)

    preds = inference(model, X_test)
    precision, recall, f1 = compute_model_metrics(y_test, preds)
    print(f"Test Precision: {precision:.3f} | Recall: {recall:.3f} | F1:{f1:.3f}")

    save_model(model, encoder, lb)

    # rubric: write per-slice metrics using the helper
    slice_text = performance_on_categorical_slice(
        df=test,
        feature="education",
        model=model,
        encoder=encoder,
        lb=lb,
        categorical_features=cat_features,
        label=label,
    )
    with open("slice_output.txt", "w", encoding="utf-8") as f:
        f.write(slice_text)

if __name__ == "__main__":
    main()
