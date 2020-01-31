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
        self.bvresults = {}
        self.INDEX = np.arange(len(sampleSizes))
        self.sampleSizes = sampleSizes
        self.evaluate = ""
        print("DGP:", dgp.__name__)

    def simulate(self, method, simulationNum = 1000, evaluate="R2"):
        print("Simulate", method.__name__)
        self.evaluate = evaluate
        meanScores = []
        mean_bias_var = {'loss':[],'bias':[],'var':[]}
        for size in self.sampleSizes:
            print("Sample size:", size)
            # Generate test set
            # X_test, y_test = self.dgp(random_state = size, n=round(size*0.2))
            X_test, y_test = self.dgp(random_state = size, n=round(100))
            scores = []
            bias_var = {'loss':[],'bias':[],'var':[]}
            for iteration in range(simulationNum):
                if simulationNum > 1:
                    print("Iteration=", iteration + 1)
                # Generate data
                X, y = self.dgp(random_state=iteration, n=size)
                # fit model
                model = method(X, y, random_state=iteration)
                # Add fitted model to list
                self.models.append(model)
                # Bias-variance decpmposition
                avg_loss, avg_bias, avg_var = bias_var_decomp(model, X_test, y_test, dgp=self.dgp)
                # Evaluate
                if evaluate=="R2":
                    score = model.score(X_test, y_test)
                elif evaluate=="RSS":
                    score = mean_squared_error(y_test, model.predict(X_test))

                # Add the scores to the list
                scores.append(score)
                for bv_key in bias_var.keys():
                    bias_var[bv_key].append(locals()['avg_'+bv_key])
            # Add mean RSS for the respective size
            meanScores.append(np.mean(scores))
            for mbv_key in bias_var.keys():
                mean_bias_var[mbv_key].append(np.mean(bias_var[mbv_key]))
        # add to results for given class name
        self.results[method.__name__] = meanScores
        self.bvresults[method.__name__] = mean_bias_var

    def plot(self, title="Score for different sample sizes", filePath=""):
        labels = []
        for name, result in self.results.items():
            plt.plot(self.sampleSizes, result)
            labels.append(name)

        plt.legend(labels, loc='upper right')
        plt.xlabel('Sample Size')
        plt.ylabel('Score')
        plt.title(title)
        if filePath:
            plt.savefig(filePath)
        else:
            plt.show()
        plt.clf()

    def bvplot(self, title="", filePath="", logScale=False):
        ax = plt.subplot(111)
        for _, result in self.bvresults.items():
            for pbv_key in result.keys():
                ax.plot(self.sampleSizes, pbv_key, data=result)
        if logScale:
            ax.set_yscale('log')
        plt.legend()
        plt.xlabel('Sample Size')
        plt.ylabel('Error')
        plt.title(title)

        if filePath:
            plt.savefig(filePath)
        else:
            plt.show()
        plt.clf()

    def bar(self, title="Score for different sample sizes", filePath="", logScale=False):
        labels = []
        index = self.INDEX
        barWidth = 0.3
        ax = plt.subplot(111)
        for name, result in self.results.items():

            ax.bar(index + barWidth, result, barWidth,
                    label=name)
            if logScale:
                ax.set_yscale('log')
            index = index + barWidth
            labels.append(name)

        labelPosition = self.INDEX + ((len(self.results) + 1)*barWidth)/2
        plt.xticks(labelPosition, self.sampleSizes)
        if len(labels) > 1:
            plt.legend(labels, loc='upper right')
        plt.xlabel('Sample Size')
        plt.ylabel(self.evaluate)
        plt.title(title)
        # ax.set_xlabel('Sample Size')
        # ax.set_ylabel('Score')
        # ax.set_title(title)
        # ax.legend()

        if filePath:
            plt.savefig(filePath)
        else:
            plt.show()
        plt.clf()


def nonLinearDGP(random_state, beta=[0.3, 5, 10, 15], n=1000):
    """ y = beta_0 + beta_1*I(x1 >= 0, x2 >= 0) + beta_2*I(x1 >= 0, x2 < 0) + beta_3*I(x1 < 0) + e """
    np.random.seed(random_state)

    mu, sigma = 0, 3 # mean and standard deviation



    eps = np.random.normal(mu, 1, size=n)
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

    mu, sigma = 0, 3 # mean and standard deviation
    eps = np.random.normal(mu, 1, size=n)
    X = pd.DataFrame( np.random.normal(mu, sigma, size=(n, 3)), columns=('x1', 'x2', 'x3') )

    y = (beta[0]
         + beta[1] * X['x1']
         + beta[2] * X['x2']
         + beta[3] * X['x3']
         + eps)

    return X, y

def nonLinearDGP_pure(X, beta=[0.3, 5, 10, 15]):
    """True DGP function for
    y = beta_0 + beta_1*I(x1 >= 0, x2 >= 0) + beta_2*I(x1 >= 0, x2 < 0) + beta_3*I(x1 < 0) + e
    """
    y = (beta[0]
         + beta[1] * X.apply(lambda x: float(x['x1'] >= 0 and x['x2'] >= 0), axis=1)
         + beta[2] * X.apply(lambda x: float(x['x1'] >= 0 and x['x2'] < 0), axis=1)
         + beta[3] * X.apply(lambda x: float(x['x1'] < 0), axis=1))
    return y

def linearDGP_pure(X, beta=[0.3, 5, 10, 15]):
    """True DGP for
     y = beta_0 + beta_1*x1 + beta_2*x2 + beta_3*x3 + e
     """
    y = beta[0] + beta[1] * X['x1'] + beta[2] * X['x2'] + beta[3] * X['x3']
    return y

def bias_var_decomp(model, X_test, y_test, dgp):
    y_pred = model.predict(X_test)
    loss = np.mean((y_pred - y_test)**2)
    if dgp == linearDGP:
        y_true = linearDGP_pure(X_test)
    elif dgp == nonLinearDGP:
        y_true = nonLinearDGP_pure(X_test)
    bias = np.mean((y_true - y_pred)**2)
    resid = np.mean(y_true - y_test)**2
    var = loss - bias - resid
    return loss, bias, var

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
    # Compare RSS of random forest and ols on increasing sample sizes from non-linear DGP
    mcs = MonteCarloSimulation(nonLinearDGP, sampleSizes = [100, 500, 1000, 5000, 10000, 50000, 75000, 100000])

    mcs.simulate(method=randomForestCV, simulationNum = 100, evaluate="RSS")
    mcs.bvplot(title="Bias-Variance Decomposition for Different Sample Sizes\nNon-Linear DGP",
                filePath="plots/bias_var_nonlinearDGP")
    mcs.simulate(method=linearRegression, simulationNum = 100, evaluate="RSS")

    mcs.bar(title="RSS for non-linear DGP",
            filePath="plots/forest_vs_ols_nonlinearDGP")


    # Compare RSS of random forest and ols on increasing sample sizes from linear DGP
    mcs = MonteCarloSimulation(linearDGP, sampleSizes = [100, 500, 1000, 5000, 10000, 50000, 75000, 100000])

    mcs.simulate(method=randomForestCV, simulationNum = 100, evaluate="RSS")
    mcs.bvplot(title="Bias-Variance Decomposition for Different Sample Sizes\nLinear DGP",
        filePath="plots/bias_var_linearDGP", logScale=True)
    mcs.simulate(method=linearRegression, simulationNum = 100, evaluate="RSS")

    mcs.bar(title=f"RSS for linear DGP",
            filePath=f"plots/forest_vs_ols_linearDGP",
            logScale=True)
