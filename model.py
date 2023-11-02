import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

pd.set_option('display.max_columns', None)
data = pd.read_csv("HW2_data.csv", usecols=['CHURN', 'tenure', 'monthlycharges', 'PAPERLESSBILLING',
                                            'DEPENDENTS', 'internetservice', 'FIBER', 'DSL', 'phoneservice'])

df_data = pd.DataFrame(data)
df_data.loc[df_data["internetservice"] == "DSL", "internetservice"] = 0
df_data.loc[df_data["internetservice"] == "Fiber optic", "internetservice"] = 1
df_data.loc[df_data["internetservice"] == "No", "internetservice"] = -1

df_data.loc[df_data["phoneservice"] == "Yes", "phoneservice"] = 1
df_data.loc[df_data["phoneservice"] == "No", "phoneservice"] = 0

df_data.loc[df_data["FIBER"] == "Yes", "FIBER"] = 1
df_data.loc[df_data["FIBER"] == "No", "FIBER"] = 0

df_data.loc[df_data["DSL"] == "Yes", "DSL"] = 1
df_data.loc[df_data["DSL"] == "No", "DSL"] = 0
# split features and labels
feature_cols = ['tenure', 'monthlycharges', 'PAPERLESSBILLING',
                                            'DEPENDENTS', 'internetservice', 'FIBER', 'DSL', 'phoneservice']
X = df_data[feature_cols]
Y = df_data.CHURN

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
logreg = LogisticRegression(max_iter=200)
model = logreg.fit(X_train, Y_train)
print("Feature Coefs: ")
print(model.coef_)
print("Model Intercept")
print(model.intercept_)
y_predicted = model.predict(X_test)
cnf_matrix = metrics.confusion_matrix(Y_test, y_predicted)
recall = metrics.recall_score(Y_test, y_predicted)
accuracy = metrics.accuracy_score(Y_test, y_predicted)
precision = metrics.precision_score(Y_test, y_predicted)
f1 = metrics.f1_score(Y_test, y_predicted)
print("Confusion Matrix for cut-off=0.5")
print(cnf_matrix)
print("Recall for cut-off=0.5")
print(recall)
print("Accuracy for cut-off=0.5")
print(accuracy)
print("Precision for cut-off=0.5")
print(precision)
print("F1 for training")
print(f1)


# change the threshold probability to 0.7
y_predicted_prob = model.predict_proba(X_test)
y_predicted_class = y_predicted_prob[:, 1]
y_predicted_updated = [1 if prob >= 0.7 else 0 for prob in y_predicted_class]
cnf_matrix = metrics.confusion_matrix(Y_test, y_predicted_updated)
recall = metrics.recall_score(Y_test, y_predicted_updated)
accuracy = metrics.accuracy_score(Y_test, y_predicted_updated)
precision = metrics.precision_score(Y_test, y_predicted_updated)
print("Confusion Matrix for cut-off=0.7")
print(cnf_matrix)
print("Recall for cut-off=0.7")
print(recall)
print("Accuracy for cut-off=0.7")
print(accuracy)
print("Precision for cut-off=0.7")
print(precision)



