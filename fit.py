"""
This script is for fitting the model to the data
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import data_setup

# Get the data
X_train, X_test, y_train, y_test = data_setup.import_train_test(data_dir="Steel_industry.csv",
                                                                test_size=0.2)

# Intialize the model
model = RandomForestClassifier()

# Fit the model
model.fit(X_train, y_train)

# Make predictions
preds = model.predict(X_test)

# Calculate different metrics
accuracy = accuracy_score(y_true=y_test,
                          y_pred=preds) * 100

f1 = f1_score(y_true=y_test,
              y_pred=preds,
              average='macro') * 100

recall = recall_score(y_true=y_test,
                      y_pred=preds, 
                      average='macro') * 100

precision = precision_score(y_true=y_test,
                            y_pred=preds, 
                            average='macro') * 100

# Print the metrics
print(f"Accuracy: {accuracy:.2f}%")
print(f"Recall: {recall:.3f}%")
print(f"Precision: {precision:.3f}%")
print(f"f1_Score: {f1:.3f}%")