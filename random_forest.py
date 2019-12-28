import os
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.ensemble import RandomForestClassifier
import feature_process_helper
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scikitplot as skplt

plt.rcParams['figure.figsize'] = [20, 10]
pd.options.mode.chained_assignment = None


# Load data & transform variables
print("Load data")
data = pd.read_csv('data/titanic.csv')
X = data[data.columns.difference(["Survived"])]
y = data["Survived"]
del data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
del X, y 

# Extract length and title of name
print("Prepare data")
X_train, X_test = feature_process_helper.names(X_train, X_test)

# Replace missing age values with means grouped  by title and class
X_train, X_test = feature_process_helper.age_impute(X_train, X_test)

# Take first letter of cabin
X_train, X_test = feature_process_helper.cabin(X_train, X_test)

# Fill missing values for embarked
X_train, X_test = feature_process_helper.embarked_impute(X_train, X_test)

# Determine family size
X_train, X_test = feature_process_helper.fam_size(X_train, X_test)

# Extract ticket length
X_train['Ticket_Len'] = X_train['Ticket'].apply(lambda x: len(x))
X_test['Ticket_Len'] = X_test['Ticket'].apply(lambda x: len(x))

# Create dummy variable for several columns
X_train, X_test = feature_process_helper.dummies(X_train, X_test, columns = ['Pclass', 'Sex', 'Embarked',
                                                                     'Cabin_Letter', 'Name_Title', 'Fam_Size'])

X_train, X_test = feature_process_helper.drop(X_train, X_test, bye = ['Ticket'])

# Tune hyper-parameters
forest = RandomForestClassifier(max_features='auto',
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1)

param_grid = { "criterion"   : ["gini", "entropy"],
               "min_samples_leaf" : [1,5,10],
               "min_samples_split" : [2, 4, 10, 12, 16],
               "n_estimators": [50, 100, 400, 700, 1000]}

param_search = GridSearchCV(estimator=forest,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=3,
                  n_jobs=-1)

param_search = param_search.fit(X_train, y_train)


print("Best Accuracy:", param_search.best_score_)
print("Best Parameters:", param_search.best_params_)


# Fit model
forest = RandomForestClassifier(**param_search.best_params_,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
forest.fit(X_train, y_train)

print("Out-of-Bag Error:", "%.4f" % forest.oob_score_)

# Plot feature importances
feat_importances = pd.Series(forest.feature_importances_, index=X_train.columns).sort_values(ascending=True)
feat_importances.nlargest(10).plot(kind='barh')
plt.title("Random Forest Feature Importances (MDI)")
plt.savefig("plots/feature_importances")
plt.clf()

# Plot confusion matrix
y_pred = forest.predict(X_test)

pred_labels = ["Died" if group==0 else "Survived"  for group in y_pred]
test_labels = ["Died" if group==0 else "Survived"  for group in y_test]

ax = skplt.metrics.plot_confusion_matrix(test_labels, pred_labels, normalize=True,
                                         title="Confusion matrix")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.savefig("plots/confusion_matrix")
