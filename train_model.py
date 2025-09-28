import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    train_model,
    inference,
    compute_model_metrics,
    save_model,   # provided by the starter
)

def main():
    # 1) Load data
    data = pd.read_csv("data/census.csv")

    # 2) Census categorical features + target
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
    label = "salary"  # '<=50K' / '>50K'

    # 3) Split
    train, test = train_test_split(
        data, test_size=0.20, random_state=42, stratify=data[label]
    )

    # 4) Process (fit on train; reuse encoder/lb on test)
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label=label,
        training=True,
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # 5) Train
    model = train_model(X_train, y_train)

    # 6) Evaluate
    preds = inference(model, X_test)
    precision, recall, f1 = compute_model_metrics(y_test, preds)
    print(f"Test Precision: {precision:.3f} | Recall: {recall:.3f} | F1:{f1:.3f}")

    # 7) Save model + encoders
    save_model(model, encoder, lb)

    # 8) Slice metrics (rubric): by education -> slice_output.txt
    slice_feature = "education"
    with open("slice_output.txt", "w", encoding="utf-8") as f:
        for val in sorted(test[slice_feature].dropna().unique()):
            df_slice = test[test[slice_feature] == val]
            if df_slice.empty:
                continue
            X_s, y_s, _, _ = process_data(
                df_slice,
                categorical_features=cat_features,
                label=label,
                training=False,
                encoder=encoder,
                lb=lb,
            )
            preds_s = inference(model, X_s)
            sp, sr, sf = compute_model_metrics(y_s, preds_s)
            f.write(
                f"{slice_feature}={val} -> "
                f"P:{sp:.3f} R:{sr:.3f} F1:{sf:.3f} N:{len(df_slice)}\n"
            )

if __name__ == "__main__":
    main()
