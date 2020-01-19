import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
import scikitplot as skplt
import matplotlib.pyplot as plt
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

adaboost = AdaBoostClassifier(random_state=1)
paramGrid = {"n_estimators": [50, 100, 400, 700, 1000],
             "learning_rate": [0.01+x/100 for x in range(100)],
             "n_estimators": [50, 100, 400, 700, 1000]}


gs = GridSearchCV(estimator=adaboost,
                  param_grid=paramGrid,
                  scoring='accuracy',
                  cv=3,
                  n_jobs=-1)


gs = gs.fit(X_train, y_train)

print("Best Accuracy:", gs.best_score_)
print("Best Parameters:", gs.best_params_)

# Fit model
adaboost = AdaBoostClassifier(**gs.best_params_, random_state=1)
adaboost.fit(X_train, y_train)

featImportances = pd.Series(adaboost.feature_importances_, index=X_train.columns).sort_values(ascending=True)
featImportances.nlargest(10).plot(kind='barh')
plt.title("Adaboost Feature Importances (MDI)")
plt.savefig("plots/feature_importances_adaboost")
plt.clf()

y_pred = adaboost.predict(X_test)
print("Accuracy:", adaboost.score(X_test, y_test))

pred_labels = ["Died" if group == 0 else "Survived" for group in y_pred]
test_labels = ["Died" if group == 0 else "Survived" for group in y_test]

ax = skplt.metrics.plot_confusion_matrix(test_labels, pred_labels,
                                         normalize=True,
                                         title="Confusion matrix")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.savefig("plots/confusion_matrix_adaboost")
