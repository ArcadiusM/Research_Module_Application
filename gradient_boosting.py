import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import scikitplot as skplt
import clean_titanic_data as clean

plt.rcParams['figure.figsize'] = [20, 10]
pd.options.mode.chained_assignment = None

# Load data & transform variables
print("Load data")
data = pd.read_csv('data/titanic.csv')
X = data[data.columns.difference(["Survived"])]
y = data["Survived"]
del data

X_train, X_test, y_train, y_test = clean.clean_titanic_data(x=X, y=y)

# Tune hyper-parameters
gradient_boosting = GradientBoostingClassifier(max_features='auto',
                                               random_state=1
                                               )

paramGrid = {"loss": ["deviance", "exponential"],
             "min_samples_leaf": [1, 5, 10],
             "min_samples_split": [2, 4, 10, 12, 16],
             "n_estimators": [50, 100, 400, 700, 1000]}

gs = GridSearchCV(estimator=gradient_boosting,
                  param_grid=paramGrid,
                  scoring='accuracy',
                  cv=3,
                  n_jobs=-1)

gs = gs.fit(X_train, y_train)

print("Best Accuracy:", gs.best_score_)
print("Best Parameters:", gs.best_params_)

# Fit model
gradient_boosting = GradientBoostingClassifier(**gs.best_params_,
                                               max_features='auto',
                                               random_state=1,
                                               )
gradient_boosting.fit(X_train, y_train)

# Plot feature importances
featImportances = pd.Series(gradient_boosting.feature_importances_, index=X_train.columns).sort_values(ascending=True)
featImportances.nlargest(10).plot(kind='barh')
plt.title("Gradient Boosting Feature Importances")
plt.savefig("plots/feature_importances_gradient_boosting")
plt.clf()

# Plot confusion matrix
y_pred = gradient_boosting.predict(X_test)
print("Accuracy:", gradient_boosting.score(X_test, y_test))

pred_labels = ["Died" if group == 0 else "Survived" for group in y_pred]
test_labels = ["Died" if group == 0 else "Survived" for group in y_test]

ax = skplt.metrics.plot_confusion_matrix(test_labels, pred_labels,
                                         normalize=True,
                                         title="Confusion matrix")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.savefig("plots/confusion_matrix_gradient_boosting")
