
import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

st.set_page_config(page_title="Lung Cancer Classification", layout="wide", initial_sidebar_state="expanded")

MODEL_DIRECTORY = "models"
TARGET_COLUMN = "LUNG_CANCER"

FEATURE_COLUMNS = ["GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE", "CHRONIC DISEASE", "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL CONSUMING", "COUGHING", "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "CHEST PAIN"]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Gaussian Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl"
}

SCALED_MODELS = {"Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine"}


@st.cache_resource
def load_artifacts():
    models = {model_name: joblib.load(os.path.join(MODEL_DIRECTORY, file_name)) for model_name, file_name in MODEL_FILES.items()}
    feature_scaler = joblib.load(os.path.join(MODEL_DIRECTORY, "feature_scaler.pkl"))
    label_encoders = joblib.load(os.path.join(MODEL_DIRECTORY, "label_encoders.pkl"))
    return models, feature_scaler, label_encoders


def prepare_features(uploaded_data, label_encoders):
    uploaded_data = uploaded_data.copy()
    uploaded_data.columns = uploaded_data.columns.str.strip()

    missing_columns = [column for column in FEATURE_COLUMNS if column not in uploaded_data.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    features = uploaded_data[FEATURE_COLUMNS].copy()

    for column in features.select_dtypes(include=["object", "string"]).columns:
        features[column] = features[column].astype(str).str.strip()

    for column in features.columns:
        if column in label_encoders:
            try:
                features[column] = label_encoders[column].transform(features[column])
            except ValueError as error:
                expected_values = ", ".join(map(str, label_encoders[column].classes_))
                raise ValueError(f"Invalid value in '{column}'. Expected values: {expected_values}") from error

    try:
        features = features.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("The uploaded dataset contains invalid feature values.") from error

    return features


def calculate_metrics(actual_values, predicted_values, predicted_probabilities, model_classes, positive_label):
    positive_probability_index = list(model_classes).index(positive_label)

    metrics = {
        "Accuracy": accuracy_score(actual_values, predicted_values),
        "AUC": roc_auc_score(actual_values, predicted_probabilities[:, positive_probability_index]),
        "Precision": precision_score(actual_values, predicted_values, pos_label=positive_label, zero_division=0),
        "Recall": recall_score(actual_values, predicted_values, pos_label=positive_label, zero_division=0),
        "F1 Score": f1_score(actual_values, predicted_values, pos_label=positive_label, zero_division=0),
        "MCC": matthews_corrcoef(actual_values, predicted_values)
    }

    return metrics


def display_metrics(metrics):
    metric_columns = st.columns(3)

    for index, (metric_name, metric_value) in enumerate(metrics.items()):
        metric_columns[index % 3].metric(metric_name, f"{metric_value:.4f}")


st.title("Lung Cancer Classification")
st.write("Upload patient test data to generate predictions and evaluate performance across trained classification models.")

with st.expander("Project Overview & Original Dataset Description", expanded=False):
    overview_column, dataset_column = st.columns(2)

    with overview_column:
        st.markdown("### Problem Statement")
        st.write(
            "In this project, I have developed and evaluated five supervised machine learning models to predict lung cancer "
            "using patient details, lifestyle habits, and symptoms. The main objective is to compare these models and identify "
             "which model performs better based on different evaluation metrics."
)

    with dataset_column:
        st.markdown("### Dataset Summary")
        st.write("• **Source**: Public Kaggle / UCI Machine Learning Repository")
        st.write("• **Total Instances**: 1,157 patient records (644 NO, 513 YES)")
        st.write("• **Target Variable**: `LUNG_CANCER` (Binary: YES / NO)")
        st.write("• **Hold-out Test Set**: 242 records (never exposed during model training)")
        st.write("• **Data Leakage Prevention**: StratifiedGroupKFold on unique symptom profiles")

    st.markdown("### Feature Attributes (15 Predictors)")
    features_col1, features_col2 = st.columns(2)

    with features_col1:
        st.write("**1. GENDER**: Patient biological sex (M / F)")
        st.write("**2. AGE**: Patient age (20 to 87 years)")
        st.write("**3. SMOKING**: Active smoking history (1 = No, 2 = Yes)")
        st.write("**4. YELLOW_FINGERS**: Yellow discoloration on fingers (1 = No, 2 = Yes)")
        st.write("**5. ANXIETY**: Patient reporting frequent anxiety (1 = No, 2 = Yes)")
        st.write("**6. PEER_PRESSURE**: Exposure to peer smoking pressure (1 = No, 2 = Yes)")
        st.write("**7. CHRONIC DISEASE**: History of chronic lung/systemic disease (1 = No, 2 = Yes)")
        st.write("**8. FATIGUE**: Ongoing fatigue and lethargy (1 = No, 2 = Yes)")

    with features_col2:
        st.write("**9. ALLERGY**: Known respiratory or seasonal allergies (1 = No, 2 = Yes)")
        st.write("**10. WHEEZING**: Audible whistling or rattling breathing (1 = No, 2 = Yes)")
        st.write("**11. ALCOHOL CONSUMING**: Regular alcohol consumption (1 = No, 2 = Yes)")
        st.write("**12. COUGHING**: Persistent or severe coughing (1 = No, 2 = Yes)")
        st.write("**13. SHORTNESS OF BREATH**: Dyspnea during regular activities (1 = No, 2 = Yes)")
        st.write("**14. SWALLOWING DIFFICULTY**: Dysphagia or painful swallowing (1 = No, 2 = Yes)")
        st.write("**15. CHEST PAIN**: Recurring chest pain or tightness (1 = No, 2 = Yes)")

with st.sidebar:
    st.title("Project Overview")
    st.write("Development and Evaluation of a Supervised Classification System Using Five Machine Learning Algorithms for Early Lung Cancer Detection")
    st.divider()

    st.markdown("**Dataset Summary**")
    st.write("• **Total Records**: 1,157 instances")
    st.write("• **Input Features**: 15 clinical predictors")
    st.write("• **Target**: `LUNG_CANCER` (YES / NO)")
    st.write("• **Evaluation Split**: 242 hold-out records")
    st.divider()

    st.markdown("**Available Models**")
    for model_name in MODEL_FILES:
        st.write(f"• {model_name}")

try:
    models, feature_scaler, label_encoders = load_artifacts()
except FileNotFoundError:
    st.error(f"Saved model files are missing in '{MODEL_DIRECTORY}' directory.")
    st.stop()
except Exception as error:
    st.error(f"Unable to load model artifacts: {error}")
    st.stop()


st.subheader("Model Prediction & Evaluation")
st.caption("By default, evaluation metrics and predictions are displayed for the benchmark test dataset (test_data.csv). You can also upload a custom CSV below.")

col_upload, col_select = st.columns([3, 2])
with col_upload:
    uploaded_file = st.file_uploader("Upload Custom Test Dataset (CSV)", type=["csv"], help="Optional: Upload your own CSV containing the 15 features to evaluate.")

with col_select:
    selected_model_name = st.selectbox("Select Classification Model", options=list(MODEL_FILES.keys()))


if uploaded_file is not None:
    uploaded_data = pd.read_csv(uploaded_file)
    data_source_label = f"Uploaded file: {uploaded_file.name}"
    is_custom_upload = True
elif os.path.exists("test_data.csv"):
    uploaded_data = pd.read_csv("test_data.csv")
    data_source_label = "Default Hold-out Test Dataset (test_data.csv)"
    is_custom_upload = False
else:
    uploaded_data = None
    data_source_label = None
    is_custom_upload = False


if uploaded_data is not None:
    try:
        uploaded_data.columns = uploaded_data.columns.str.strip()

        if uploaded_data.empty:
            raise ValueError("The dataset is empty.")

        features = prepare_features(uploaded_data, label_encoders)
        selected_model = models[selected_model_name]

        if selected_model_name in SCALED_MODELS:
            model_input = feature_scaler.transform(features)
        else:
            model_input = features

        encoded_predictions = selected_model.predict(model_input)
        target_encoder = label_encoders[TARGET_COLUMN]
        predicted_labels = target_encoder.inverse_transform(encoded_predictions.astype(int))

        prediction_results = pd.DataFrame({"Predicted LUNG_CANCER": predicted_labels})
        actual_values = None

        if TARGET_COLUMN in uploaded_data.columns:
            actual_labels = uploaded_data[TARGET_COLUMN].astype(str).str.strip()
            actual_values = target_encoder.transform(actual_labels)
            prediction_results.insert(0, "Actual LUNG_CANCER", actual_labels.values)

        if is_custom_upload:
            st.success(f"Active Dataset: **{data_source_label}** ({len(uploaded_data):,} records) | Model: **{selected_model_name}**")
        else:
            st.info(f"Active Dataset: **{data_source_label}** ({len(uploaded_data):,} records) | Model: **{selected_model_name}**")

        prediction_tab, evaluation_tab, comparison_tab, data_tab = st.tabs(["Predictions", "Model Evaluation", "All Models Comparison", "Dataset Preview"])

        with prediction_tab:
            st.subheader("Prediction Results")
            st.dataframe(prediction_results, use_container_width=True, hide_index=True)

            prediction_counts = prediction_results["Predicted LUNG_CANCER"].value_counts().rename_axis("Predicted Class").reset_index(name="Records")
            count_column, chart_column = st.columns(2)

            with count_column:
                st.markdown("**Prediction Summary**")
                st.dataframe(prediction_counts, use_container_width=True, hide_index=True)

            with chart_column:
                st.markdown("**Class Distribution**")
                st.bar_chart(prediction_counts.set_index("Predicted Class"))

            csv_output = prediction_results.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions", data=csv_output, file_name="lung_cancer_predictions.csv", mime="text/csv")

        with evaluation_tab:
            if actual_values is None:
                st.info("Include the LUNG_CANCER target column in your CSV to calculate evaluation metrics.")
            else:
                st.subheader("Evaluation Metrics")

                predicted_probabilities = selected_model.predict_proba(model_input)
                positive_label = int(target_encoder.transform(["YES"])[0])

                metrics = calculate_metrics(actual_values, encoded_predictions, predicted_probabilities, selected_model.classes_, positive_label)
                display_metrics(metrics)

                matrix_column, report_column = st.columns(2)
                class_labels = list(range(len(target_encoder.classes_)))

                with matrix_column:
                    st.markdown("**Confusion Matrix**")
                    matrix = confusion_matrix(actual_values, encoded_predictions, labels=class_labels)
                    figure, axis = plt.subplots(figsize=(5, 4))

                    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=target_encoder.classes_, yticklabels=target_encoder.classes_, cbar=False, ax=axis)

                    axis.set_xlabel("Predicted Label")
                    axis.set_ylabel("Actual Label")
                    figure.tight_layout()
                    st.pyplot(figure, use_container_width=True)
                    plt.close(figure)

                with report_column:
                    st.markdown("**Classification Report**")
                    report = classification_report(actual_values, encoded_predictions, labels=class_labels, target_names=target_encoder.classes_, output_dict=True, zero_division=0)
                    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

        with comparison_tab:
            st.subheader("All Models Evaluation & Benchmark")
            st.caption("Side-by-side performance comparison of all five trained classifiers on the uploaded dataset.")

            if actual_values is None:
                st.info("Upload a dataset with the LUNG_CANCER column to view metrics comparison across all models.")
                st.markdown("**Predictions from All Models**")
                all_predictions = pd.DataFrame()
                for m_name, model in models.items():
                    m_input = feature_scaler.transform(features) if m_name in SCALED_MODELS else features
                    all_predictions[m_name] = target_encoder.inverse_transform(model.predict(m_input).astype(int))
                st.dataframe(all_predictions, use_container_width=True)
            else:
                comparison_rows = []
                positive_label = int(target_encoder.transform(["YES"])[0])

                for m_name, model in models.items():
                    m_input = feature_scaler.transform(features) if m_name in SCALED_MODELS else features
                    m_preds = model.predict(m_input)
                    m_probs = model.predict_proba(m_input)
                    m_metrics = calculate_metrics(actual_values, m_preds, m_probs, model.classes_, positive_label)
                    comparison_rows.append({
                        "ML Model Name": m_name,
                        "Accuracy": m_metrics["Accuracy"],
                        "AUC": m_metrics["AUC"],
                        "Precision": m_metrics["Precision"],
                        "Recall": m_metrics["Recall"],
                        "F1 Score": m_metrics["F1 Score"],
                        "MCC": m_metrics["MCC"]
                    })

                comparison_df = pd.DataFrame(comparison_rows)

                display_df = comparison_df.copy()
                for col in ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]:
                    display_df[col] = display_df[col].map(lambda val: f"{val:.4f}")

                st.markdown("**Model Evaluation Comparison Table**")
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.markdown("**Accuracy by Model**")
                    st.bar_chart(comparison_df.set_index("ML Model Name")["Accuracy"])

                with chart_col2:
                    st.markdown("**F1 Score by Model**")
                    st.bar_chart(comparison_df.set_index("ML Model Name")["F1 Score"])

                best_model_row = comparison_df.sort_values(by="F1 Score", ascending=False).iloc[0]
                st.success(f"Top Performing Model: **{best_model_row['ML Model Name']}** (Accuracy: {best_model_row['Accuracy']:.4f}, F1: {best_model_row['F1 Score']:.4f}, Recall: {best_model_row['Recall']:.4f}, AUC: {best_model_row['AUC']:.4f})")

                with st.expander("Model Performance Observations"):
                    st.markdown("""
                    - **Support Vector Machine (SVM)**: Achieves the highest Accuracy, Recall, and F1 Score. Minimizes False Negatives, which is critical in clinical screening.
                    - **K-Nearest Neighbors (KNN)**: Achieved the highest AUC-ROC, showing excellent ranking and class separation on standardized features.
                    - **Decision Tree**: Balanced non-linear baseline capturing symptom combinations without requiring scaling.
                    - **Gaussian Naive Bayes**: Fast probabilistic model with good recall, but lower AUC due to conditional independence assumption among correlated symptoms.
                    - **Logistic Regression**: High precision baseline, but lower recall due to linear decision boundary limits on complex symptom combinations.
                    """)

        with data_tab:
            st.subheader("Uploaded Test Dataset")
            st.write(f"Rows: {uploaded_data.shape[0]:,} | Columns: {uploaded_data.shape[1]}")
            st.dataframe(uploaded_data.head(20), use_container_width=True)

    except Exception as error:
        st.error(f"Unable to process the uploaded CSV: {error}")

else:
    st.warning("No dataset found. Please upload a CSV containing the 15 feature columns.")

