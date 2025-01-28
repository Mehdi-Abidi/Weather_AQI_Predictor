import os
from math import sqrt
import hopsworks
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


import numpy as np
from scipy.stats import zscore

def fetch_historical_data():
    project = hopsworks.login(api_key_value=os.getenv("HW_API_KEY"))
    fs = project.get_feature_store()
    feature_group_name = "air_quality_features_f_cleaned"
    feature_group_version = 1

    try:
        # Retrieve data from the feature group
        feature_group = fs.get_feature_group(name=feature_group_name, version=feature_group_version)
        historical_data = feature_group.read()
        print("Fetched historical data successfully!")
        return historical_data
    except Exception as e:
        print(f"Failed to fetch historical data: {e}")
        return None

def correlation(dataset, threshold):
    col_corr = set()
    numeric_df = dataset.select_dtypes(include=['int64', 'float64'])
    corr_matrix = numeric_df.corr()

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) < threshold:
                colname = corr_matrix.columns[i]
                if colname != 'future_aqi':
                    col_corr.add(colname)
    return col_corr

def save_to_hopsworks(data, project, feature_group_name, version, description):
    try:
        fs = project.get_feature_store()
        feature_group = fs.get_feature_group(name=feature_group_name, version=version)
        feature_group.delete()
        feature_group = fs.get_or_create_feature_group(
            name=feature_group_name,
            version=version,
            primary_key=["id"],
            description=description
        )
        feature_group.insert(data)
        print(f"Data saved to feature group: {feature_group_name}")
    except Exception as e:
        print(f"Failed to save data: {e}")

historical_data = fetch_historical_data()

if historical_data is not None:
    # Remove outliers based on z-scores
    historical_data = historical_data[(np.abs(zscore(historical_data[['pm25', 'pm10']])) < 3).all(axis=1)]

    project = hopsworks.login(api_key_value=os.getenv("HW_API_KEY"))
    rows_with_nan = historical_data[historical_data["future_aqi"].isna()]
    # print(rows_with_nan)
    rows_without_nan = historical_data.dropna(subset=["future_aqi"]).dropna()

    # Separate rows with NaN values in the target column
    # rows_with_nan = historical_data[historical_data["future_aqi"].isna()]
    # historical_data.dropna(inplace=True)

    # Save rows with NaN to a new feature group
    save_to_hopsworks(
        rows_with_nan,
        project,
        "air_quality_nan_rows",
        version=1,
        description="Rows with NaN values for future AQI prediction"
    )

    # df1 = historical_data.copy()
    # corrfeatures = correlation(df1, 0.06)  
    # df1.drop(columns=corrfeatures, inplace=True)

    # if df1 is not None:
        # Preprocessing

    X = rows_without_nan.drop(["future_aqi", "id"], axis=1, errors="ignore")
    y = rows_without_nan["future_aqi"]
    
    # X = df1.drop(["future_aqi", "id"], axis=1, errors="ignore")  # Drop target and ID
    # y = df1["future_aqi"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

    # Train Logistic Regression Model
    linear_model = LogisticRegression()
    linear_model.fit(X_train, y_train)

    train_predictions = linear_model.predict(X_train)
    test_predictions = linear_model.predict(X_test)

    # Evaluate on the test set
    accuracy = accuracy_score(y_test, test_predictions)
    precision = precision_score(y_test, test_predictions, average="weighted")  # Weighted for multiclass
    recall = recall_score(y_test, test_predictions, average="weighted")
    f1 = f1_score(y_test, test_predictions, average="weighted")

    # Display metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Classification report (optional, for a detailed view per class)
    print("\nClassification Report:")
    print(classification_report(y_test, test_predictions))

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, test_predictions)
    print(cm)
# else:
#     print("No data available for training.")




# Model #2
import os
from math import sqrt
import hopsworks
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import numpy as np

if historical_data is not None:
    project = hopsworks.login(api_key_value=os.getenv("HW_API_KEY"))

    X = rows_without_nan.drop(["future_aqi", "id"], axis=1, errors="ignore")
    y = rows_without_nan["future_aqi"]

    # X = df1.drop(["future_aqi", "id"], axis=1, errors="ignore")
    # y = df1["future_aqi"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

    alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    # Store results
    results = []

    # Iterate through each alpha and fit the Ridge model
    for alpha in alphas:
        model = RidgeClassifier(alpha=alpha)
        model.fit(X_train, y_train)

        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        # Evaluate on the test set
        print("Evaluation for alpha value : ", alpha)
        print("\n\n")
        accuracy = accuracy_score(y_test, test_predictions)
        precision = precision_score(y_test, test_predictions, average="weighted")  # Weighted for multiclass
        recall = recall_score(y_test, test_predictions, average="weighted")
        f1 = f1_score(y_test, test_predictions, average="weighted")

        # Display metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Classification report (optional, for a detailed view per class)
        print("\nClassification Report:")
        print(classification_report(y_test, test_predictions))

        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, test_predictions)
        print(cm)
else:
    print("No data available for training.")




# Model 3

import hopsworks
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from math import sqrt
import numpy as np

if historical_data is not None:
    X = rows_without_nan.drop(["future_aqi", "id"], axis=1, errors="ignore")
    y = rows_without_nan["future_aqi"]
    # X = df1.drop(["future_aqi", "id"], axis=1, errors="ignore")  # Drop target and ID
    # y = df1["future_aqi"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

    # Initialize the Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=300,  # Number of trees in the forest
        max_depth=8,    # Maximum depth of each tree
        random_state=42    # Seed for reproducibility
    )

    # Train the model
    rf_model.fit(X_train, y_train)

    # Predict on test data
    train_predictions = rf_model.predict(X_train)
    test_predictions = rf_model.predict(X_test)

    # Evaluate on the test set
    accuracy = accuracy_score(y_test, test_predictions)
    precision = precision_score(y_test, test_predictions, average="weighted")  # Weighted for multiclass
    recall = recall_score(y_test, test_predictions, average="weighted")
    f1 = f1_score(y_test, test_predictions, average="weighted")

    # Display metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Classification report (optional, for a detailed view per class)
    print("\nClassification Report:")
    print(classification_report(y_test, test_predictions))

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, test_predictions)
    print(cm)
else:
    print("No data available for training.")






# Storing Models into the Model Registry

import os
import joblib

# Specify the model directory
model_dir = "models"
# Create directories if they don't exist
os.makedirs(model_dir, exist_ok=True)

import hopsworks
project = hopsworks.login(api_key_value=os.getenv("HW_API_KEY"))
mr = project.get_model_registry()

#exporting logistic regression
model_path = os.path.join(model_dir, "linear_model.json")
joblib.dump(linear_model, model_path)

#exporting ridge regression model to model dir
model_path = os.path.join(model_dir, "rr_model.json")
joblib.dump(model, model_path)

#exporting random forest regression model to model dir
model_path = os.path.join(model_dir, "rf_model.json")
joblib.dump(rf_model, model_path)

model_dir = "models"
model_files = [f for f in os.listdir(model_dir) if f.endswith(".json")]

for model_file in model_files:
    model_path = os.path.join(model_dir, model_file)
    model_name = os.path.splitext(model_file)[0]

    # Createing and register the model in Hopsworks

    model = mr.python.create_model(
        name=model_name,
        description=f"Model {model_name} for AQI prediction"
    )

    # Upload the model file to the Model Registry

    model.save(model_path)
    print(f"Registered and uploaded model: {model_name}")
