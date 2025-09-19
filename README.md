📝 Project Description – Heart Disease ML Pipeline

This project implements a comprehensive end-to-end Machine Learning pipeline on the Heart Disease UCI dataset to analyze, predict, and visualize heart disease risk factors.

It covers every stage of a real-world data science workflow:

Data Preprocessing & Cleaning:
Handle missing values, encode categorical variables, scale numerical features, and conduct exploratory data analysis (EDA) using histograms, boxplots, and correlation heatmaps.

Dimensionality Reduction (PCA):
Reduce feature space while retaining most variance, determine optimal components, and visualize cumulative variance.

Feature Selection:
Apply feature importance (Random Forest/XGBoost), Recursive Feature Elimination (RFE), and Chi-Square tests to choose the best predictors.

Supervised Learning Models:
Train and evaluate Logistic Regression, Decision Trees, Random Forest, and Support Vector Machines (SVM) on an 80/20 train-test split.
Metrics include Accuracy, Precision, Recall, F1-Score, ROC and AUC.

Unsupervised Learning Models:
Apply K-Means and Hierarchical Clustering (dendrogram analysis) to discover hidden patterns and compare clusters with actual labels.

Model Optimization:
Use GridSearchCV and RandomizedSearchCV to tune hyperparameters and improve performance.

Deployment:
Export the final model pipeline as a .pkl file.
Build a Streamlit web app to allow real-time prediction from user input and show visualizations.
Deploy the app publicly with Ngrok for instant access.

Version Control:
Full project hosted on GitHub, including notebooks, scripts, trained models, and documentation.

⚙️ Key Tools & Libraries

Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn

PCA, RFE, Chi-Square tests

Logistic Regression, Decision Trees, Random Forest, SVM

K-Means, Hierarchical Clustering

GridSearchCV, RandomizedSearchCV

Streamlit, Ngrok, GitHub

🎯 Deliverables

Cleaned dataset & EDA visuals

PCA and feature-selection outputs

Trained supervised & unsupervised models with metrics

Optimized model saved in .pkl format

Streamlit UI for live predictions

Public link (Ngrok) and GitHub repository
