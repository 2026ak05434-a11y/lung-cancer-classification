# Lung Cancer Classification - Machine Learning Assignment 2

## a. Problem Statement

The objective of this project is to build and compare different machine learning classification models to predict whether a person has lung cancer based on their age, gender, and reported symptoms.

In this project, I implemented five supervised machine learning algorithms, evaluated their performance using different classification metrics, and deployed the trained models through a Streamlit web application.

## b. Dataset Description

**Dataset Source:** Public Kaggle / UCI Machine Learning Repository (Lung Cancer Dataset)

The dataset contains 1,157 records and 15 input features. The target column is `LUNG_CANCER`, which contains two possible values: `YES` and `NO`.

The input features include:

1. `GENDER` - Gender of the person.
2. `AGE` - Age of the person.
3. `SMOKING` - Smoking status.
4. `YELLOW_FINGERS` - Presence of yellow fingers.
5. `ANXIETY` - Anxiety level.
6. `PEER_PRESSURE` - Peer pressure.
7. `CHRONIC DISEASE` - Presence of chronic disease.
8. `FATIGUE` - Presence of fatigue.
9. `ALLERGY` - Presence of allergies.
10. `WHEEZING` - Presence of wheezing.
11. `ALCOHOL CONSUMING` - Alcohol consumption status.
12. `COUGHING` - Presence of coughing.
13. `SHORTNESS OF BREATH` - Breathing difficulty.
14. `SWALLOWING DIFFICULTY` - Difficulty in swallowing.
15. `CHEST PAIN` - Presence of chest pain.

The dataset contains 15 input features, which exceeds the minimum requirement of 12 features specified in the assignment.

### Data Preprocessing and Splitting

During data analysis, I identified duplicate records with identical feature combinations. To avoid having the same feature profiles in both training and testing datasets, I used `StratifiedGroupKFold` with 5 folds and `random_state=42`.

The grouping was performed using the input feature combinations. This helped prevent identical feature profiles from appearing in both datasets.

The final hold-out test dataset contains 242 records:

- NO: 144 records
- YES: 98 records

I also applied feature scaling using `StandardScaler` for Logistic Regression, KNN, and SVM. The scaler was fitted using only the training data and then applied to the test data.

## c. GitHub Repository and Live Application

**GitHub Repository:**  
https://github.com/2026ak05434-a11y/lung-cancer-classification

**Live Streamlit Application:**  
https://lung-cancer-classification.streamlit.app

The GitHub repository contains the notebook, trained models, preprocessing artifacts, test dataset, and Streamlit application source code.

## d. Models Used and Performance Comparison

I implemented and evaluated the following five machine learning classification algorithms using the same training and testing datasets:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN, k=5)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM with RBF kernel)

### Model Comparison

The following table shows the performance of all five models based on the test dataset.

| ML Model Name          |  Accuracy  |  AUC   | Precision  |   Recall   |     F1     |    MCC     |
| :--------------------- | :--------: | :----: | :--------: | :--------: | :--------: | :--------: |
| Logistic Regression    |   0.8884   | 0.9453 |   0.9610   |   0.7551   |   0.8457   |   0.8038   |
| Decision Tree          |   0.9174   | 0.9329 |   0.9535   |   0.8367   |   0.8913   |   0.8347   |
| KNN                    |   0.9256   | 0.9622 |   0.9651   |   0.8469   |   0.9022   |   0.8473   |
| Naive Bayes            |   0.9091   | 0.8950 |   0.9130   |   0.8571   |   0.8842   |   0.8106   |
| Support Vector Machine | **0.9380** | 0.9580 | **0.9663** | **0.8776** | **0.9198** | **0.8722** |

### Model Performance Observations

**Logistic Regression**

Logistic Regression provides a good baseline with 88.84% accuracy. It achieves high precision, but its recall is lower than the other models. This indicates that it misses more positive cases compared to the other classifiers.

**Decision Tree**

The Decision Tree model achieves 91.74% accuracy. It can identify different feature combinations using decision rules and does not require feature scaling. Its overall performance is better than Logistic Regression.

**K-Nearest Neighbors**

KNN achieves 92.56% accuracy and the highest AUC of 0.9622 among the five models. Standardizing the features helps KNN compare records based on their distances more effectively.

**Gaussian Naive Bayes**

Naive Bayes achieves 90.91% accuracy and 85.71% recall. It provides good performance, although its assumption that features are conditionally independent may not always hold for this dataset.

**Support Vector Machine**

SVM with an RBF kernel achieves the highest accuracy of 93.80%. It also gives the highest precision, recall, F1 score, and MCC among the five models. The RBF kernel helps the model capture non-linear relationships between the input features.

### Overall Model Selection

**Best Performing Model: Support Vector Machine (RBF Kernel)**

Based on the evaluation results, SVM provides the best overall performance on the hold-out test dataset.

The main reasons for selecting SVM are:

1. It achieves the highest accuracy of 93.80%.
2. It has the highest recall of 87.76%, meaning it identifies more positive cases than the other evaluated models.
3. It achieves the highest F1 score of 0.9198 and MCC of 0.8722.
4. Its AUC of 0.9580 indicates strong ability to distinguish between the two classes.

Therefore, SVM is the best-performing model in this experiment based on the selected evaluation metrics.

## Repository Structure

```text
project-folder/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── saved_models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── svm.pkl
│   ├── feature_scaler.pkl
│   └── label_encoders.pkl
└── ML_Assignment_2.ipynb
```

## Running the Application Locally

Follow these steps to run the Streamlit application on your local system.

### 1. Clone the Repository

```bash
git clone https://github.com/2026ak05434-a11y/lung-cancer-classification.git
cd lung-cancer-classification
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Streamlit Application

```bash
streamlit run app.py
```

### 4. Use the Application

1. Open the Streamlit application in your browser.
2. Upload the `test_data.csv` file.
3. Select one of the five available classification models.
4. Click the **Run Prediction** button.
5. View the prediction results and model evaluation metrics.
6. Check the confusion matrix and classification report.
7. Download the prediction results as a CSV file.

