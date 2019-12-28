import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split



class MonteCarloSimulation():
    def __init__(self, dgp, sampleSizes):
        self.dgp = dgp
        self.models = []
        self.meanScores = []
        self.results = {}
        self.sampleSizes = sampleSizes
        print(f"DGP: {dgp.__name__}")

    def simulate(self, method, simulationNum = 1000, evaluate="R2"):
        print(f"Simulate {method.__name__}")
        meanScores = []
        for size in self.sampleSizes:
            print("Sample size:", size)
            # Generate test set
            X_test, y_test = self.dgp(random_state = size, n=round(size*0.2))
            scores = []
            for iteration in range(simulationNum):
                print("Iteration=", iteration + 1)
                # Generate data
                X, y = self.dgp(random_state = iteration, n=size)
                # fit model
                model = method(X,y, random_state=iteration)
                # Add fitted model to list
                self.models.append(model)
                # Evaluate
                if evaluate=="R2":
                    score = model.score(X_test, y_test)
                elif evaluate=="RSS":
                    score = mean_squared_error(y_test, model.predict(X_test))

                # Add the scores to the list
                scores.append(score)
            # Add mean RSS for the respective size
            meanScores.append(np.mean(scores))        
        # add to results for given class name
        self.results[method.__name__] = meanScores

    def plot(self, title="Score for different sample sizes", filePath=""):
        labels = []
        for name, result in self.results.items():
            plt.plot(self.sampleSizes, result)
            labels.append(name)

        plt.legend(labels, loc='upper left')
        plt.xlabel('Sample Size')
        plt.ylabel('Score')
        plt.title(title)
        if filePath:
            plt.savefig(filePath)
        else:
            plt.show()
        plt.clf()


def nonLinearDGP(random_state, beta=[0.3, 5, 10, 15], n=1000):
    """ y = beta_0 + beta_1*I(x1 >= 0, x2 >= 0) + beta_2*I(x1 >= 0, x2 < 0) + beta_3*I(x1 < 0) + e """
    np.random.seed(random_state)
    
    mu, sigma = 0, 2 # mean and standard deviation
    
    

    eps = np.random.normal(mu, 2, size=n)
    X = pd.DataFrame( np.random.normal(mu, sigma, size=(n, 2)), columns=('x1', 'x2') )
    y = (beta[0] 
         + beta[1] * X.apply(lambda x: float(x['x1'] >= 0 and x['x2'] >= 0), axis=1) 
         + beta[2] * X.apply(lambda x: float(x['x1'] >= 0 and x['x2'] < 0), axis=1) 
         + beta[3] * X.apply(lambda x: float(x['x1'] < 0), axis=1) 
         + eps)
    
    return X, y


def linearDGP(random_state, beta=[0.3, 5, 10, 15], n=1000):
    """ y = beta_0 + beta_1*x1 + beta_2*x2 + beta_3*x3 + e """
    np.random.seed(random_state)
    
    mu, sigma = 0, 2 # mean and standard deviation
    eps = np.random.normal(mu, 2, size=n)
    X = pd.DataFrame( np.random.normal(mu, sigma, size=(n, 3)), columns=('x1', 'x2', 'x3') )
    
    y = (beta[0] 
         + beta[1] * X['x1']
         + beta[2] * X['x2']
         + beta[3] * X['x3']
         + eps)
    
    return X, y    


def randomForestCV(features, target, n_estimators=10, random_state = 101, regression=True):
    # Find the best parameters for the model
    parameterGrid = {
        'n_estimators': [i + 1 for i in range(n_estimators)]
    }
    if regression:
        forest = RandomForestRegressor(random_state=random_state)
    else:
        forest = RandomForestClassifier(random_state=random_state)
        
    gridForest = GridSearchCV(estimator=forest, param_grid=parameterGrid, cv = 5)
    gridForest.fit(features, target)
    
    param = {"random_state": 0}
    param = {**param, **gridForest.best_params_}
    if regression:
        forestOptim = RandomForestRegressor(**param).fit(features, target)
    else:
        forestOptim = RandomForestClassifier(**param).fit(features, target)
    
    return forestOptim    


def linearRegression(features, target, random_state):
    ols = LinearRegression().fit(features, target)
    return ols

if __name__ == '__main__':
    # Iterate through DGPs 
    for dgp in [linearDGP, nonLinearDGP]:
        mcs = MonteCarloSimulation(dgp, sampleSizes = [100, 1000, 5000, 10000])

        mcs.simulate(method=randomForestCV, simulationNum = 5, evaluate="RSS")

        mcs.simulate(method=linearRegression, simulationNum = 5, evaluate="RSS")
        
        mcs.plot(title=f"RSS-Scores for a {dgp.__name__}",
                 filePath=f"plots/forest_vs_ols_{dgp.__name__}")

    