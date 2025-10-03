import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

def train_model(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf

def compute_model_metrics(y, preds):
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    return precision, recall, fbeta

def inference(model, X):
    return model.predict(X)

def save_model(model, encoder, lb, out_dir="model"):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "model.joblib"))
    joblib.dump(encoder, os.path.join(out_dir, "encoder.joblib"))
    joblib.dump(lb, os.path.join(out_dir, "label_binarizer.joblib"))

def load_model(out_dir: str = "model"):
    model = joblib.load(os.path.join(out_dir, "model.joblib"))
    encoder = joblib.load(os.path.join(out_dir, "encoder.joblib"))
    lb = joblib.load(os.path.join(out_dir, "label_binarizer.joblib"))
    return model, encoder, lb

# ---- rubric helper: performance_on_categorical_slice ----
def performance_on_categorical_slice(
    df,
    feature: str,
    model,
    encoder,
    lb,
    categorical_features,
    label: str,
):
    """
    Compute precision/recall/F1 for each value of a categorical feature.
    Returns a newline-joined string ready to write to slice_output.txt
    """
    from ml.data import process_data  # local import to avoid circulars

    lines = []
    for val in sorted(df[feature].dropna().unique()):
        slice_df = df[df[feature] == val]
        if slice_df.empty:
            continue
        X_s, y_s, _, _ = process_data(
            slice_df,
            categorical_features=categorical_features,
            label=label,
            training=False,
            encoder=encoder,
            lb=lb,
        )
        preds = inference(model, X_s)
        p, r, f1 = compute_model_metrics(y_s, preds)
        lines.append(f"{feature}={val} -> P:{p:.3f} R:{r:.3f} F1:{f1:.3f} N:{len(slice_df)}")
    return "\n".join(lines) + "\n"
