# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import streamlit as st

# Load the Dataset
file_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
column_names = ["ID", "Diagnosis"] + ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']
data = pd.read_csv(file_path, header=None, names=column_names)

# Preprocessing and Data Cleaning
data.drop('ID', axis=1, inplace=True)  # Dropping ID column as it's not needed for prediction
data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})  # Encoding the Diagnosis (M=1, B=0)
X = data.drop('Diagnosis', axis=1)  # Features
y = data['Diagnosis']  # Target variable

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Applying SMOTE to balance the data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# AdaBoost Classifier Model
ada_classifier = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1), 
    n_estimators=50,
    learning_rate=1.0,
    algorithm="SAMME",
    random_state=42
)
ada_classifier.fit(X_train_smote, y_train_smote)
y_pred_smote = ada_classifier.predict(X_test)

# Performance Metrics
accuracy_smote = accuracy_score(y_test, y_pred_smote)
precision_smote = precision_score(y_test, y_pred_smote)
recall_smote = recall_score(y_test, y_pred_smote)
f1_smote = f1_score(y_test, y_pred_smote)


# Best Model from GridSearch
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.1, 0.5, 1.0, 1.5],
    'estimator__max_depth': [1, 2, 3]
}
grid_search = GridSearchCV(estimator=ada_classifier, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
grid_search.fit(X_train_smote, y_train_smote)
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# ROC Curve and AUC
y_pred_proba_best = best_model.predict_proba(X_test)[:, 1]
fpr_best, tpr_best, thresholds_best = roc_curve(y_test, y_pred_proba_best)
roc_auc_best = auc(fpr_best, tpr_best)

# Streamlit Layout
st.title('Breast Cancer Diagnosis Prediction')
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Make Prediction"])

# Home Page: Overview and Data Visualizations
if page == "Home":
    st.subheader("Basic Overview of the Dataset")
    st.write(data.head())  # Display the first few rows of the dataset
    st.write(f"Shape of the dataset: {data.shape}")  # Display dataset dimensions
    
    # Pie chart for the distribution of diagnosis
    diagnosis_counts = data['Diagnosis'].value_counts()
    st.subheader("Distribution of Breast Cancer Diagnosis")
    fig, ax = plt.subplots()
    ax.pie(diagnosis_counts, labels=['Benign (0)', 'Malignant (1)'], autopct='%1.1f%%', startangle=90, colors=['skyblue', 'salmon'])
    ax.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.
    st.pyplot(fig)
    
    # Correlation Heatmap
    st.subheader("Feature Correlation Heatmap")
    plt.figure(figsize=(20,12))
    sns.heatmap(data.corr(),annot=True,fmt='.0%')
    plt.title('Feature Correlation Heatmap')
    plt.show()
    
    # Positive Correlation Scatter Plots
    st.subheader("Scatter Plots of Positively Correlated Features")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.scatterplot(x=data['perimeter_mean'], y=data['radius_worst'], hue=data['Diagnosis'], palette=['blue', 'red'], ax=axes[0, 0])
    axes[0, 0].set_title('Perimeter Mean vs Radius Worst')
    
    sns.scatterplot(x=data['area_mean'], y=data['radius_worst'], hue=data['Diagnosis'], palette=['blue', 'red'], ax=axes[0, 1])
    axes[0, 1].set_title('Area Mean vs Radius Worst')
    
    sns.scatterplot(x=data['texture_mean'], y=data['texture_worst'], hue=data['Diagnosis'], palette=['blue', 'red'], ax=axes[1, 0])
    axes[1, 0].set_title('Texture Mean vs Texture Worst')
    
    sns.scatterplot(x=data['area_mean'], y=data['radius_worst'], hue=data['Diagnosis'], palette=['blue', 'red'], ax=axes[1, 1])
    axes[1, 1].set_title('Area Mean vs Radius Worst')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Outlier Detection and Visualization
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))
    num_outliers_per_column = outliers.sum()
    
    st.subheader("Outlier Visualization")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=num_outliers_per_column.index, y=num_outliers_per_column.values)
    ax.set_xticklabels(num_outliers_per_column.index, rotation=90)
    st.pyplot(fig)

    # Confusion Matrix for Best AdaBoost Model
    st.subheader("Confusion Matrix for Best AdaBoost Classifier")
    y_pred_best = best_model.predict(X_test)  # Using the best model from GridSearchCV
    conf_matrix_best = confusion_matrix(y_test, y_pred_best)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(conf_matrix_best, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix (Best Model)')
    st.pyplot(fig)

    # Classification Report for Best Model
    st.subheader("Classification Report for Best Model")
    report = classification_report(y_test, y_pred_best, target_names=["Benign", "Malignant"], output_dict=True)

    # Convert the classification report dictionary to a DataFrame
    report_df = pd.DataFrame(report).transpose()

    # Display the DataFrame in Streamlit as a table
    st.dataframe(report_df)
    


# Make Prediction Page: User Input for Prediction
elif page == "Make Prediction":
    st.subheader("Make Predictions")
    st.write("Enter the values for the features to predict whether the tumor is malignant or benign.")
    
    # Create input fields for user to enter feature values
    user_inputs = {}
    for feature in X.columns:
        user_inputs[feature] = st.number_input(f"Enter value for {feature}", min_value=0.0, value=1.0)

    # Predict if the tumor is malignant or benign
    if st.button("Make Prediction"):
        # Scale the user input values using the same scaler as used in training
        user_input_array = np.array(list(user_inputs.values())).reshape(1, -1)
        user_input_scaled = scaler.transform(user_input_array)
        
        # Predict using the trained model
        prediction = best_model.predict(user_input_scaled)
        
        if prediction == 1:
            st.write("The tumor is *Malignant*.")
        else:
            st.write("The tumor is *Benign*.")