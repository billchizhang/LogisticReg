import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

pd.set_option('display.max_columns', None)
data = pd.read_csv("HW2_data.csv", usecols=['CHURN', 'tenure', 'monthlycharges', 'PAPERLESSBILLING',
                                            'DEPENDENTS', 'internetservice', 'FIBER', 'DSL', 'phoneservice'])

df_data = pd.DataFrame(data)

# set up dummy variables
df_data.loc[df_data["internetservice"] == "DSL", "internetservice"] = 0
df_data.loc[df_data["internetservice"] == "Fiber optic", "internetservice"] = 1
df_data.loc[df_data["internetservice"] == "No", "internetservice"] = -1

df_data.loc[df_data["phoneservice"] == "Yes", "phoneservice"] = 1
df_data.loc[df_data["phoneservice"] == "No", "phoneservice"] = 0


# adding features
df_data["tenure_half"] = df_data["tenure"] / 2
df_data["tenure_double"] = df_data["tenure"] * 2
df_data["tenure_half_DSL"] = df_data["tenure_half"] * df_data["DSL"]
df_data["tenure_half_FIBER"] = df_data["tenure_half"] * df_data["FIBER"]
df_data["tenure_double_DSL"] = df_data["tenure_double"] * df_data["DSL"]
df_data["tenure_double_FIBER"] = df_data["tenure_double"] * df_data["FIBER"]

# split features and labels
''' feature_cols = ['tenure', 'monthlycharges', 'PAPERLESSBILLING',
                                            'DEPENDENTS', 'internetservice', 'FIBER', 'DSL', 'phoneservice']'''
feature_cols = ['DSL', 'FIBER', 'tenure', 'monthlycharges']
X = df_data[feature_cols]
Y = df_data.CHURN
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
logreg = LogisticRegression().fit(X_train, Y_train)

print("Feature Coefs: ")
print(logreg.coef_)
print("Model Intercept")
print(logreg.intercept_)

# using training set
y_predicted = logreg.predict(X_train)

cnf_matrix = metrics.confusion_matrix(Y_train, y_predicted)
recall = metrics.recall_score(Y_train, y_predicted)
accuracy = metrics.accuracy_score(Y_train, y_predicted)
precision = metrics.precision_score(Y_train, y_predicted)
f1 = metrics.f1_score(Y_train, y_predicted)
print("Confusion Matrix for training")
print(cnf_matrix)
print("Recall for training")
print(recall)
print("Accuracy for training")
print(accuracy)
print("Precision for training")
print(precision)
print("F1 for training")
print(f1)

y_predicted = logreg.predict(X_test)
# using testing set
cnf_matrix = metrics.confusion_matrix(Y_test, y_predicted)
recall = metrics.recall_score(Y_test, y_predicted)
accuracy = metrics.accuracy_score(Y_test, y_predicted)
precision = metrics.precision_score(Y_test, y_predicted)
f1 = metrics.f1_score(Y_test, y_predicted)
print("Confusion Matrix for test")
print(cnf_matrix)
print("Recall for test")
print(recall)
print("Accuracy for test")
print(accuracy)
print("Precision for test")
print(precision)
print("F1 for test")
print(f1)

