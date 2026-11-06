import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
)

st.set_page_config(page_title="Lung Cancer Classification", layout="wide")

MODEL_DIRECTORY = "saved_models"
TARGET_COLUMN = "LUNG_CANCER"

FEATURE_COLUMNS = [
    "GENDER",
    "AGE",
    "SMOKING",
    "YELLOW_FINGERS",
    "ANXIETY",
    "PEER_PRESSURE",
    "CHRONIC DISEASE",
    "FATIGUE",
    "ALLERGY",
    "WHEEZING",
    "ALCOHOL CONSUMING",
    "COUGHING",
    "SHORTNESS OF BREATH",
    "SWALLOWING DIFFICULTY",
    "CHEST PAIN",
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Gaussian Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl",
}


@st.cache_resource
def load_artifacts():
    models = {
        model_name: joblib.load(os.path.join(MODEL_DIRECTORY, file_name))
        for model_name, file_name in MODEL_FILES.items()
    }
    feature_scaler = joblib.load(os.path.join(MODEL_DIRECTORY, "feature_scaler.pkl"))
    label_encoders = joblib.load(os.path.join(MODEL_DIRECTORY, "label_encoders.pkl"))
    return models, feature_scaler, label_encoders


def prepare_features(uploaded_data, label_encoders):
    uploaded_data = uploaded_data.copy()
    uploaded_data.columns = uploaded_data.columns.str.strip()

    missing_columns = [column for column in FEATURE_COLUMNS if column not in uploaded_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required feature columns: {', '.join(missing_columns)}")

    features = uploaded_data[FEATURE_COLUMNS].copy()

    for column in features.select_dtypes(include=["object", "string"]).columns:
        features[column] = features[column].astype(str).str.strip()

    for column in features.columns:
        if column in label_encoders:
            try:
                features[column] = label_encoders[column].transform(features[column])
            except ValueError as error:
                raise ValueError(
                    f"Unknown value found in '{column}'. Expected values: "
                    f"{', '.join(label_encoders[column].classes_)}"
                ) from error

    for column in features.columns:
        features[column] = pd.to_numeric(features[column], errors="raise")

    return features


def calculate_metrics(actual_values, predicted_values, positive_label, predicted_probabilities, positive_probability_index):
    metrics = {
        "Accuracy": accuracy_score(actual_values, predicted_values),
        "AUC": roc_auc_score(actual_values, predicted_probabilities[:, positive_probability_index]),
        "Precision": precision_score(actual_values, predicted_values, pos_label=positive_label, zero_division=0),
        "Recall": recall_score(actual_values, predicted_values, pos_label=positive_label, zero_division=0),
        "F1 Score": f1_score(actual_values, predicted_values, pos_label=positive_label, zero_division=0),
        "MCC": matthews_corrcoef(actual_values, predicted_values),
    }
    return metrics


st.title("Lung Cancer Classification")
st.write("Upload a test CSV file and select a trained machine learning model to view predictions and evaluation results.")

try:
    models, feature_scaler, label_encoders = load_artifacts()
except FileNotFoundError as error:
    st.error("Saved model files were not found. Run the model-saving cells in your notebook and place the saved_models folder beside app.py.")
    st.stop()

uploaded_file = st.file_uploader("Upload test CSV file", type=["csv"])

selected_model_name = st.selectbox("Select classification model", list(MODEL_FILES.keys()))

if uploaded_file is not None:
    try:
        uploaded_data = pd.read_csv(uploaded_file)
        uploaded_data.columns = uploaded_data.columns.str.strip()

        st.subheader("Uploaded data")
        st.write(f"Rows: {uploaded_data.shape[0]} | Columns: {uploaded_data.shape[1]}")
        st.dataframe(uploaded_data.head())

        features = prepare_features(uploaded_data, label_encoders)
        selected_model = models[selected_model_name]

        if selected_model_name in ["Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine"]:
            model_input = feature_scaler.transform(features)
        else:
            model_input = features

        encoded_predictions = selected_model.predict(model_input)
        target_encoder = label_encoders[TARGET_COLUMN]
        predicted_labels = target_encoder.inverse_transform(encoded_predictions.astype(int))

        st.subheader("Predictions")
        prediction_results = pd.DataFrame({"Predicted LUNG_CANCER": predicted_labels})

        if TARGET_COLUMN in uploaded_data.columns:
            prediction_results.insert(0, "Actual LUNG_CANCER", uploaded_data[TARGET_COLUMN].astype(str).str.strip().values)

        st.dataframe(prediction_results)

        if TARGET_COLUMN in uploaded_data.columns:
            actual_values = target_encoder.transform(uploaded_data[TARGET_COLUMN].astype(str).str.strip())
            predicted_probabilities = selected_model.predict_proba(model_input)
            positive_label = int(target_encoder.transform(["YES"])[0])
            positive_probability_index = list(selected_model.classes_).index(positive_label)

            metrics = calculate_metrics(
                actual_values,
                encoded_predictions,
                positive_label,
                predicted_probabilities,
                positive_probability_index,
            )

            st.subheader("Model Evaluation")
            metric_columns = st.columns(3)
            for index, (metric_name, metric_value) in enumerate(metrics.items()):
                metric_columns[index % 3].metric(metric_name, f"{metric_value:.4f}")

            st.subheader("Confusion Matrix")
            confusion = confusion_matrix(actual_values, encoded_predictions, labels=[0, 1])
            figure, axis = plt.subplots()
            sns.heatmap(
                confusion,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=target_encoder.classes_,
                yticklabels=target_encoder.classes_,
                ax=axis,
            )
            axis.set_xlabel("Predicted label")
            axis.set_ylabel("Actual label")
            st.pyplot(figure)
            plt.close(figure)

            st.subheader("Classification Report")
            report = classification_report(
                actual_values,
                encoded_predictions,
                target_names=target_encoder.classes_,
                output_dict=True,
                zero_division=0,
            )
            st.dataframe(pd.DataFrame(report).transpose().round(4))

        else:
            st.info("The uploaded CSV does not contain LUNG_CANCER. Predictions are shown without evaluation metrics.")

        csv_output = prediction_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download predictions",
            data=csv_output,
            file_name="lung_cancer_predictions.csv",
            mime="text/csv",
        )

    except Exception as error:
        st.error(f"Unable to process the uploaded CSV: {error}")
else:
    st.info("Please upload a CSV file containing the 15 input features. Include LUNG_CANCER if you also want to view evaluation metrics.")
