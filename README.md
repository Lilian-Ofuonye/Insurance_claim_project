## Motor Insurance Claim Prediction – Ultimate Claim Cost Modelling
### All codes and model development are embedded in python file named "Week1 New"
### Project Overview

This project builds a machine learning model to predict Ultimate Claim Amount for motor insurance claims using structured claims, policyholder, and third-party data.

The objective is to:
* Improve reserving accuracy at FNOL (First Notification of Loss)
* Understand key drivers of claim severity
* Provide explainable predictions suitable for insurance decision-making
The final champion model is XGBoost, selected based on strong predictive performance and interpretability using SHAP.

### Dataset Description
The modelling dataset was constructed from three separate CSV files:
* Claims dataset
* Policyholders dataset
* Third-party dataset
#### Schema Design
* Each claim is the modelling unit.
* The third-party dataset was aggregated to claim level to avoid duplicate claim rows
* Aggregation included:
  * Number of third parties
  * Total injuries
  * Maximum injury severity
All datasets were merged into a single structured modelling table.

### Data Cleaning & Preparation
#### Integrity Checks
* Verified unique keys before merging
* Removed duplicates
* Assessed and handled missing values
* Converted date fields to datetime format
* Checked for impossible values (e.g. negative vehicle age)
#### Feature Engineering
Created meaningful insurance features:
* FNOL_Delay_Days – reporting delay
* Accident_Month / Accident_Day_of_Week – seasonality
* Age_Minus_Experience – proxy for age at first driving
* Late_Notification_Flag – behavioural risk signal

### Outlier Handling
* Annual_Mileage: Winsorised using IQR method
* FNOL Delay: Flagged rather than clipped (extreme delays may be informative)
* Ultimate Claim Amount:
  * Heavily right-skewed
  * Log-transformed using log1p to stabilise variance

### Target used in modelling:
* Log_Ultimate_Claim_Amount
Predictions were bias-corrected and exponentiated back to original £ scale.

### Modelling Approach
#### Baseline Model
Intercept-only model (mean prediction)
| Model    | MAE    | RMSE   |
| -------- | ------ | ------ |
| Baseline | 13,900 | 38,001 |

#### Models Trained
* Linear Regression
* Decision Tree
* Random Forest
* XGBoost

All models were:
* Trained on log-transformed target
* Evaluated on test set
* Back-transformed to original currency scale
* Compared using MAE and RMSE

#### Model Performance Comparison
| Model             | MAE       | RMSE      | Improvement vs Baseline |
| ----------------- | --------- | --------- | ----------------------- |
| Baseline          | 13,900    | 38,001    | —                       |
| Linear Regression | 4,062     | 35,345    | ~71%                    |
| Decision Tree     | 3,607     | 21,804    | ~74%                    |
| Random Forest     | 2,312     | 16,031    | ~83%                    |
| **XGBoost**       | **1,698** | **9,434** | **~88%**                |

### Champion Model: XGBoost
XGBoost delivered:
* Lowest MAE
* Lowest RMSE
* Strong generalisation
* Stable residual behaviour
* Clear feature interpretability
It significantly outperformed linear and tree-based models.

### Explainability – SHAP Analysis
SHAP was applied to the XGBoost model to provide transparent interpretation.

#### Top Drivers of Claim Severity
* Estimated Claim Amount
* Severity Band
* Claim Type (Fire)
* FNOL Delay
* Claim Complexity
#### Key Insights
* Estimated claim amount shows a strong but diminishing marginal impact.
* Severity band behaves in a clear ordered, step-wise manner.
* FNOL delay acts as a behavioural cost multiplier.
* Complexity and third-party involvement increase predicted severity.
This aligns with real-world insurance behaviour.

### Residual Diagnostics
Residual analysis showed:
* Random Forest and XGBoost significantly reduced prediction spread.
* XGBoost residuals were tightly clustered around zero.
* Some volatility remains at extreme claim values (expected in severity modelling).

### Project Structure
The notebook includes:
* Data ingestion
* Schema validation
* Aggregation logic
* Feature engineering
* Outlier handling
* Log transformation
* Model training (sklearn pipeline)
* Bias correction
* Evaluation metrics
* Residual diagnostics
* SHAP explainability

### Key Takeaways
* Log-transforming severity materially improves modelling stability.
* Behavioural features (FNOL delay) matter.
* Early financial estimate is highly predictive.
* Gradient boosting models outperform linear and bagging methods.
* Explainability is achievable even with advanced models.

### Tools & Libraries
* Python
* pandas
* numpy
* scikit-learn
* xgboost
* shap
* matplotlib / seaborn

