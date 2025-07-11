# Project-Insights-Decoding-Insurance-Premium-Prediction-

## Project Overview

This project delves into building machine learning models to predict **Insurance Premium Amount** based on various customer and policy attributes. Despite comprehensive preprocessing and the application of powerful machine learning algorithms, the analysis revealed significant challenges in predicting the target variable with the provided dataset. This README documents the entire process, the key findings, and provides insights into the limitations of the data for this specific predictive task.

## Dataset

The dataset contains information related to insurance policies and customer demographics, with the goal of predicting the `Premium Amount`.

**Dataset Source:** [https://drive.google.com/file/d/1bQ8RE4HrVakjJlWlfDmmy6OiwyYa4wdB/view]

**Key Columns include (but are not limited to):**
* `Premium Amount`: The target variable to be predicted.
* `Age`, `Number of Dependents`, `Vehicle Age`: Demographic and vehicle-related information.
* `Credit Score`, `Health Score`: Customer-specific scores.
* `Policy Start Date`, `Insurance Duration`: Time-related policy details.
* Categorical features like `Policy Type`, `Customer Feedback`, `Smoking Status`, `Exercise Frequency`, `Property Type`, etc.

## Project Structure and Workflow

The project followed a standard machine learning pipeline:

1.  **Exploratory Data Analysis (EDA):** Initial data understanding, visualization, and identification of data quality issues.
2.  **Data Preprocessing & Feature Engineering:** Cleaning, transforming, and creating new features to prepare the data for modeling.
3.  **Model Building & Evaluation (Regression):** Attempting to predict the continuous `Premium Amount`.
4.  **Model Building & Evaluation (Classification):** Pivoting to predict `Premium Categories` due to regression challenges.
5.  **Persistence:** Saving the prepared dataset for future use.
6.  **Conclusion & Recommendations:** Summarizing findings and suggesting future steps.

---

## Detailed Process

### 1. Exploratory Data Analysis (EDA)

* **Initial Data Inspection:** Checked data types, missing values, and unique values for all columns.
* **Target Variable Analysis:** `Premium Amount` was found to be highly right-skewed, necessitating transformation.
* **Feature Distribution Analysis:** Visualized distributions of numerical and categorical features. Many numerical features also exhibited significant skewness.
* **Correlation Analysis (Crucial Insight):** A **heatmap** was generated to visualize correlations between numerical features and the target variable (`Premium Amount_log` after transformation). This analysis revealed **little to no strong linear correlation between the target variable and the independent variables**. This early observation was a critical precursor to the challenges faced in modeling.

### 2. Data Preprocessing & Feature Engineering

To prepare the raw data for machine learning models, the following steps were executed:

* **Handling Missing Values:**
    * Numerical columns (`Credit Score`, `Health Score`, `Annual Income`, `Previous Claims`) were imputed using their respective **medians**.
    * Categorical columns were imputed with the **mode**.
* **Feature Engineering from `Policy Start Date`:**
    * Extracted `Policy_Start_Year`, `Policy_Start_Month`, `Policy_Start_Day`, and `Policy_Start_Day_of_Week`.
    * Calculated `Policy_Age_Years` based on the difference from a assumed current date.
* **Log Transformations (using `np.log1p`):**
    * Applied to highly skewed numerical features: `Premium Amount` (target), `Annual Income`, `Health Score`, `Previous Claims`, `Number of Vehicles`. This was crucial to normalize distributions and improve model learning.
* **Categorical Encoding:**
    * Applied **One-Hot Encoding** to all nominal categorical features to convert them into a numerical format suitable for ML models.
* **Feature Scaling:**
    * Applied **Standard Scaling** (`StandardScaler`) to all numerical features (including log-transformed ones) to ensure they contribute equally to the model, preventing features with larger scales from dominating.
* **Column Dropping:** Removed redundant or original columns after engineering/transformation (e.g., original `Policy Start Date`, `Customer ID`).
* **Dataset Persistence:** The final preprocessed DataFrame (`df_processed`) was saved as a **Parquet file** (`preprocessed_insurance_data.parquet`) for efficient future loading, avoiding repetitive preprocessing.

### 3. Model Building & Evaluation (Regression Task)

The initial goal was to predict the continuous `Premium Amount_log`.

* **Data Split:** The preprocessed data was split into training (80%) and testing (20%) sets.
* **Models Explored:**
    * **Random Forest Regressor:** A robust ensemble bagging model.
        * `n_estimators=50` yielded MAE: $684.85, RMSE: $1019.51, R2: **-0.2479**
        * `n_estimators=100` yielded MAE: $683.22, RMSE: $1019.18, R2: **-0.2471**
    * **LightGBM Regressor:** A highly efficient gradient boosting model, generally known for superior performance on tabular data.
        * `n_estimators=200` (single split) yielded MAE: $665.14, RMSE: $955.34, R2: **-0.0957**
        * `n_estimators=20` (single split) yielded MAE: $664.67, RMSE: $955.02, R2: **-0.0950**
* **Cross-Validation for LightGBM:** To ensure the robustness of the results, 5-Fold Cross-Validation was performed for LightGBM.
    * Average MAE: **$662.08 (+/- $2.90)**
    * Average RMSE: **$951.36 (+/- $4.06)**
    * Average R2: **-0.0944 (+/- 0.0016)**

**Key Finding:** The consistent and significantly **negative R-squared values** (much less than 0) across all regression models and evaluation methods indicate that **our models performed worse than simply predicting the mean `Premium Amount` for every policy**. This strongly suggested an inherent lack of predictive signal in the dataset for this task, aligning with the initial weak correlations observed in EDA.

### 4. Model Building & Evaluation (Classification Task)

Given the poor regression performance, the problem was reframed as a classification task to predict `Premium Categories`.

* **Target Creation:** `Premium Amount_log` was binned into three categories ('Low', 'Medium', 'High') based on quantiles to ensure a balanced distribution.
* **Model Used:** LightGBM Classifier.
* **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score.
* **Results:**
    * **Accuracy: 0.3356 (33.56%)**
    * Individual class precision, recall, and F1-scores were also very low (around 0.25-0.51).

**Key Finding:** An accuracy of 33.56% is barely above random guessing (33.33% for three classes). This further confirmed that the available features lack sufficient discriminative power to even categorize `Premium Amount` effectively.

### 5. Persistence

The final preprocessed dataset (`df_processed`) was saved to a Parquet file (`preprocessed_insurance_data.parquet`) to enable quick loading and avoid re-running computationally intensive preprocessing steps in future sessions.

## Conclusion & Key Learnings

This project unequivocally demonstrates that despite rigorous data preprocessing, feature engineering, and the application of highly capable machine learning models (Random Forest, LightGBM) for both regression and classification tasks, **a reliable predictive model for `Insurance Premium Amount` could not be built with the current dataset.**

The consistently **negative R-squared values** in regression and **near-random accuracy** in classification, coupled with the initial observations of weak correlations in the heatmap, strongly point to an **inherent lack of predictive signal within the existing features.** It's highly probable that the `Premium Amount` is determined by crucial factors not present in this dataset, or that the data contains overwhelming noise.

## Recommendations for Future Work

For any future attempts to predict `Insurance Premium Amount`, the focus must shift from algorithmic tuning to **significant data enhancement**:

1.  **Acquire More Comprehensive & Relevant Features:** This is paramount. Collaborate with insurance domain experts (e.g., actuaries, underwriters) to identify and integrate vital features that genuinely drive premium calculations. Examples include:
    * Detailed claims history (frequency, severity, type of past claims).
    * Specific policy coverage details, deductibles, and riders.
    * Granular geographic location data (specific addresses, local risk factors).
    * Driving records (violations, accidents).
    * More in-depth health data/medical history.
    * Information on specific underwriting rules or competitive pricing strategies.
2.  **Understand Data Generation:** If the dataset is synthetic or sampled, investigate its generation process to understand if relationships were intentionally simplified or randomized.
3.  **Explore Alternative Problem Definitions (beyond classification):** If acquiring more data for precise prediction proves infeasible, consider other analytical goals like:
    * **Customer Segmentation:** Grouping customers by their characteristics (unsupervised learning).
    * **Anomaly Detection:** Identifying policies with unusual premium amounts given their features (unsupervised learning for fraud/error detection).

This project serves as a valuable lesson in understanding the limitations of data and the critical role of domain expertise in guiding feature selection and problem framing for real-world machine learning applications.

## How to Run This Project

To replicate and explore this project, you'll need a Python environment with the following libraries:

* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`
* `seaborn`
* `lightgbm`
* `pyarrow` (for `.parquet` file handling)

You can install them using pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn lightgbm pyarrow
