from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def main():

    base_dir = Path(__file__).resolve().parent
    dataset_path = base_dir / "CCS_MapData_preprocessed.csv"

    print(f"Membaca dataset dari: {dataset_path}")

    df = pd.read_csv(dataset_path)

    print(f"Ukuran dataset: {df.shape}")
    print(f"Kolom dataset: {list(df.columns)}")

    target_col = "Overall Status"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"\nTarget: {target_col}")
    print(f"Jumlah fitur: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\nData training: {X_train.shape}")
    print(f"Data testing : {X_test.shape}")

    cat_cols = [
        "Storage and/or Capture",
        "Continent Name",
        "Country Code",
        "Project Type"
    ]

    num_cols = [
        "DOE Support",
        "Exact Checkbox",
        "Paper",
        "Regional Partnership",
        "Latitude",
        "Longitude",
        "Size_Capture_Amount"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                cat_cols
            ),
            (
                "num",
                "passthrough",
                num_cols
            )
        ]
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=42
                )
            )
        ]
    )

    mlflow.set_experiment("CCS_ML_Classification")

    mlflow.autolog()

    print("\nMemulai pelatihan model...")

    with mlflow.start_run():

        model_pipeline.fit(X_train, y_train)

        print("Pelatihan model selesai.")

        y_pred = model_pipeline.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        report = classification_report(
            y_test,
            y_pred
        )

        print("HASIL EVALUASI MODEL")

        print(f"Test Accuracy: {accuracy:.4f}")

        print("\nClassification Report:")
        print(report)

        print("Training dan logging MLflow selesai.")

if __name__ == "__main__":
    main()
