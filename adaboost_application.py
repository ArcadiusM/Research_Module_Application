import os
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import scikitplot as skplt
import clean_titanic_data as clean
import statistics
import warnings

warnings.filterwarnings("ignore")
plt.rcParams['figure.figsize'] = [20, 10]
pd.options.mode.chained_assignment = None

    
# Load data & transform variables
print("Load data")
data = pd.read_csv('data/titanic.csv')
X = data[data.columns.difference(["Survived"])]
y = data["Survived"]
del data

X_train, X_test, Y_train, Y_test = clean.clean_titanic_data(x=X, y=y)

X = X_train.append(X_test, ignore_index=True)
Y = Y_train.append(Y_test, ignore_index=True)
del X_train, X_test, Y_train, Y_test

splitNumber = 5
kf = KFold(n_splits=splitNumber, random_state=111)
accuracyResults = []
confMatrix = np.array([[0.0, 0.0],
                       [0.0, 0.0]])
fold = 1                       
for train_index, test_index in kf.split(X):
    print("Fold: ", fold)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]


    # Tune hyper-parameters
    adaboost = AdaBoostClassifier(random_state=1)
    paramGrid = {"n_estimators": [50, 100, 400, 700, 1000],
                "learning_rate": [0.01+x/100 for x in range(100)]}

    gs = GridSearchCV(estimator=adaboost,
                    param_grid=paramGrid,
                    scoring='accuracy',
                    cv=3,
                    n_jobs=-1)

    gs = gs.fit(X_train, Y_train)

    # Fit model
    adaboost = AdaBoostClassifier(**gs.best_params_, random_state=1)
    adaboost.fit(X_train, Y_train)
    # print("Accuracy:", forest.score(X_test, Y_test))
    accuracyResults.append(adaboost.score(X_test, Y_test))

    Y_pred = adaboost.predict(X_test)

    cm = confusion_matrix(Y_test, Y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    confMatrix = np.add(confMatrix, cm)
    fold += 1

confMatrix = confMatrix / float(splitNumber)

# Log accuracy result
with open("logs/adaboost_results.log", "w") as logFile:
    print(f"Mean accuracy over {splitNumber}-Folds: {statistics.mean(accuracyResults)}\n",
          file=logFile)

# Plot confusion matrix
df_cm = pd.DataFrame(confMatrix, ["Survived (True)", "Died (True)"], ["Survived (Predicted)", "Died (Predicted)"])
print(confMatrix)
sn.set(font_scale=2.5) # for label size
ax = sn.heatmap(df_cm, cmap='coolwarm', annot=True, annot_kws={"size": 45}) # font size
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title("Confusion matrix for Adaboost averaged over 5 folds")

plt.savefig("plots/confusion_matrix_adaboost")
