🧠 Stroke Risk Prediction: Clinical Data Analysis
📌 Project Overview
This project utilizes a clinical dataset of 5,110 patients to develop a machine learning model capable of predicting stroke risk. dataset : ## 📌 Project Overview
This project utilizes a clinical dataset of **5,110 patients** to predict stroke risk.
* **Data Source:** [Stroke Prediction Dataset on Kaggle](https://www.kaggle.com/code/rishabh057/healthcare-dataset-stroke-data/notebook) The primary goal was to handle significant class imbalances and missing data to create a reliable screening tool for healthcare providers.
🛠️ Data Science Workflow
I followed a structured pipeline using the tidymodels framework in R:
    Data Cleaning: Performed Median Imputation for missing BMI values to maintain the dataset's statistical distribution.

    Feature Engineering: * Normalization: Scaled numeric variables (Age, Glucose) to a uniform range.
Dummy Encoding: Converted categorical factors (Work Type, Smoking Status) into binary format.
Handling Imbalance: Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the target classes (Stroke vs. No-Stroke), ensuring the model could identify rare "at-risk" cases.

    Model Selection: Evaluated Logistic Regression and Random Forest models.
📊 Key Results

    Final Accuracy: 76.1%
    Primary Predictors: Age (highest importance), Avg Glucose Level, and Hypertension.
    Inference: The model successfully prioritizes clinical factors, providing a baseline for early medical intervention.

🚀 Technical Stack
    Language: R
    Libraries: tidyverse, tidymodels, themis (SMOTE), ranger, vip

📂 Repository Structure

    data/: Raw and cleaned datasets.
    scripts/: R scripts for cleaning, modeling, and visualization.
    plots/: Visualizations of feature importance and confusion matrices.

