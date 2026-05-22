import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# Regression Model
from sklearn.linear_model import LinearRegression

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Dataset
data = {
    "study_hours": [1,2,3,4,5,6,7,8,9,10],
    "attendance": [50,55,60,65,70,75,80,85,90,95],
    "sleep_hours": [4,5,5,6,6,7,7,8,8,9],
    "previous_marks": [40,45,50,55,60,65,70,75,80,85],
    "final_score": [35,40,45,52,60,67,72,80,88,95]
}

df = pd.DataFrame(data)

# -----------------------------
# REGRESSION PART
# -----------------------------

# Features
X = df[[
    "study_hours",
    "attendance",
    "sleep_hours",
    "previous_marks"
]]

# Target for regression
y_reg = df["final_score"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_reg,
    test_size=0.2,
    random_state=42
)

# Linear Regression Model
lr_model = LinearRegression()

# Training
lr_model.fit(X_train, y_train)

# Predictions
reg_predictions = lr_model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, reg_predictions)
mse = mean_squared_error(y_test, reg_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, reg_predictions)

print("----- LINEAR REGRESSION -----")

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# New Student
new_student = pd.DataFrame(
    [[7, 85, 8, 75]],
    columns=[
        "study_hours",
        "attendance",
        "sleep_hours",
        "previous_marks"
    ]
)

score_prediction = lr_model.predict(new_student)

print("Predicted Final Score:", score_prediction[0])

print("Coefficients:", lr_model.coef_)
print("Intercept:", lr_model.intercept_)

# -----------------------------
# CLASSIFICATION PART
# -----------------------------

# Create Pass/Fail Column
df["pass_fail"] = df["final_score"].apply(
    lambda x: 1 if x >= 50 else 0
)

# Features
X_class = df[[
    "study_hours",
    "attendance",
    "sleep_hours",
    "previous_marks"
]]

# Target for classification
y_class = df["pass_fail"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_class,
    y_class,
    test_size=0.2,
    random_state=42
)

# --------------------------------
# LOGISTIC REGRESSION
# --------------------------------

log_model = LogisticRegression()

log_model.fit(X_train, y_train)

log_predictions = log_model.predict(X_test)

print("\n----- LOGISTIC REGRESSION -----")

print("Predictions:", log_predictions)

print("Accuracy:",
      accuracy_score(y_test, log_predictions))

print("Confusion Matrix:")
print(confusion_matrix(y_test, log_predictions))

# Probability Output
print("Probability Prediction:")
print(log_model.predict_proba(new_student))

# --------------------------------
# DECISION TREE
# --------------------------------

tree_model = DecisionTreeClassifier(max_depth=3)

tree_model.fit(X_train, y_train)

tree_predictions = tree_model.predict(X_test)

print("\n----- DECISION TREE -----")

print("Predictions:", tree_predictions)

print("Accuracy:",
      accuracy_score(y_test, tree_predictions))

print("Confusion Matrix:")
print(confusion_matrix(y_test, tree_predictions))

print("Feature Importances:")
print(tree_model.feature_importances_)

# --------------------------------
# RANDOM FOREST
# --------------------------------

forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

forest_model.fit(X_train, y_train)

forest_predictions = forest_model.predict(X_test)

print("\n----- RANDOM FOREST -----")

print("Predictions:", forest_predictions)

print("Accuracy:",
      accuracy_score(y_test, forest_predictions))

print("Confusion Matrix:")
print(confusion_matrix(y_test, forest_predictions))

print("Feature Importances:")
print(forest_model.feature_importances_)

# Prediction on New Student
final_prediction = forest_model.predict(new_student)

print("\nFinal Pass/Fail Prediction:")

if final_prediction[0] == 1:
    print("PASS")
else:
    print("FAIL")